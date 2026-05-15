import argparse
import json
import os
from datetime import datetime, timezone
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version, PackageNotFoundError

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_ppl_korean, eval_downstream

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    return model

# Eval task sets. Each entry is (task_name, num_fewshot); names must match
# lm-eval-harness v0.4+ task registry.
EVAL_TASK_SETS = {
    # Proposal-stated 3-task downstream eval (strict scope, no KMMLU/MC4-ko ppl).
    # Use this bundle to stay aligned with the original Final Project Proposal.
    "proposal": [
        ("mmlu", 5),
        ("gsm8k", 8),
        ("kobest_hellaswag", 0),
    ],
    # Same as 'proposal' but skips GSM8K. GSM8K's chain-of-thought generation
    # triggers CUDA kernel failures on Wanda/SparseGPT-pruned LLaMA-3-8B
    # (numerical instability in SwiGLU MLP with 50%-zero weights). Use this
    # bundle when GSM8K causes crashes; final report documents the exclusion.
    "proposal_no_gsm8k": [
        ("mmlu", 5),
        ("kobest_hellaswag", 0),
    ],
    # KoBEST-only recovery bundle. MMLU's 56K loglikelihood requests also
    # crashes Wanda-pruned LLaMA-3-8B (same CUDA instability as GSM8K).
    # KoBEST has only ~2K requests so the smaller load avoids the crash.
    # Use this for the 4 Wanda calibration variants whose MMLU runs failed.
    "kobest_only": [
        ("kobest_hellaswag", 0),
    ],
    # Phase 3 main experiment: English knowledge, math, two Korean benchmarks.
    # KMMLU is the second Korean benchmark so the headline Korean result
    # does not rest on KoBEST-HellaSwag alone. (Extended beyond proposal.)
    "headline": [
        ("mmlu", 5),
        ("gsm8k", 8),
        ("kobest_hellaswag", 0),
        ("kmmlu", 5),
    ],
    # Phase 1 anchor reproduction: the Wanda paper's Table 8 7-task NLU bundle,
    # all 0-shot. Reproducing these numbers ties our results to the literature.
    "wanda_nlu": [
        ("boolq", 0),
        ("rte", 0),
        ("hellaswag", 0),
        ("winogrande", 0),
        ("arc_easy", 0),
        ("arc_challenge", 0),
        ("openbookqa", 0),
    ],
}


def get_task_configs(eval_tasks):
    """Resolve --eval_tasks to a list of (task, num_fewshot) tuples.

    'all' unions every named set, deduplicating by task name (first occurrence wins).
    """
    if eval_tasks == "all":
        seen = set()
        merged = []
        for tasks in EVAL_TASK_SETS.values():
            for t, n in tasks:
                if t not in seen:
                    seen.add(t)
                    merged.append((t, n))
        return merged
    return EVAL_TASK_SETS[eval_tasks]


# Wanda paper Table 2 reports the simple average across these 7 tasks.
# Phase 1 anchor reproduction compares against this average; auto-computed
# whenever the wanda_nlu bundle was run.
NLU_BUNDLE_TASKS = [
    "boolq", "rte", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa",
]


def compute_nlu_bundle_avg(results):
    """Average accuracy across the Wanda paper's 7-task NLU bundle.

    Returns None if any of the 7 tasks is missing from results or its
    accuracy key cannot be found. Tries both newer lm-eval ('acc,none')
    and older ('acc') key variants.
    """
    accs = []
    for task in NLU_BUNDLE_TASKS:
        task_block = results.get(task)
        if not task_block:
            return None
        # lm-eval returns {task_name: {metric: value, ...}, ...}
        inner = task_block.get(task)
        if not inner:
            return None
        acc = None
        for key in ("acc,none", "acc"):
            v = inner.get(key)
            if isinstance(v, (int, float)):
                acc = v
                break
        if acc is None:
            return None
        accs.append(acc)
    return sum(accs) / len(accs)


def _safe_version(pkg):
    """importlib.metadata.version with PackageNotFoundError -> None."""
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def write_manifest(save_dir, args, sparsity_ratio_actual, ppl_test, ppl_korean,
                   downstream_results, nlu_avg):
    """Write {save_dir}/manifest.json with the full reproducibility record.

    predictions.md section 7 ("forbidden moves") implicitly requires this:
    every run must be self-describing so we can later verify nothing was
    silently changed (model, seed, calibration source, library versions).
    """
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "model": args.model,
            "prune_method": args.prune_method,
            "sparsity_ratio_requested": args.sparsity_ratio,
            "sparsity_type": args.sparsity_type,
            "calib_data": args.calib_data,
            "nsamples": args.nsamples,
            "seed": args.seed,
            "use_variant": args.use_variant,
            "eval_zero_shot": args.eval_zero_shot,
            "eval_tasks": args.eval_tasks if args.eval_zero_shot else None,
            "override_shot": args.override_shot,
            "eval_korean_ppl": args.eval_korean_ppl,
            "eval_limit": args.eval_limit,
        },
        "sparsity_ratio_actual": sparsity_ratio_actual,
        "library_versions": {
            "torch": _safe_version("torch"),
            "transformers": _safe_version("transformers"),
            "accelerate": _safe_version("accelerate"),
            "datasets": _safe_version("datasets"),
            "lm_eval": _safe_version("lm_eval"),
        },
        "results": {
            "ppl_wikitext2": ppl_test,
            "ppl_mc4_ko": ppl_korean,
            "nlu_bundle_avg": nlu_avg,
            "downstream": downstream_results,
        },
    }
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--calib_data', type=str, default='c4',
                        choices=['c4', 'mc4_ko', 'the_stack', 'random'],
                        help='Calibration dataset source.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save per-layer pruning masks (~6-8 GB per run as torch.bool). Default: off.')

    parser.add_argument("--eval_zero_shot", action="store_true",
                        help="Run downstream eval (task bundle selected by --eval_tasks).")
    parser.add_argument("--eval_tasks", type=str, default="headline",
                        choices=list(EVAL_TASK_SETS.keys()) + ["all"],
                        help="Which task bundle to run when --eval_zero_shot is set. "
                             "'headline' = MMLU/GSM8K/KoBEST-HS/KMMLU (Phase 3 main). "
                             "'wanda_nlu' = the Wanda paper's 7-task NLU bundle (Phase 1 anchor). "
                             "'all' = union.")
    parser.add_argument("--eval_limit", type=int, default=None,
                        help="Limit number of examples per eval task (for fast debugging). Default: full eval.")
    parser.add_argument("--eval_batch_size", type=str, default="auto",
                        help="lm-eval-harness batch size: 'auto' (default), or an integer. "
                             "Use '1' to work around CUDA kernel failures on pruned LLaMA-3-8B.")
    parser.add_argument("--override_shot", type=int, default=None,
                        help="If set, override the canonical num_fewshot for every task in "
                             "the selected bundle. e.g. --override_shot 0 forces 0-shot for "
                             "MMLU and GSM8K. Use for Phase 3 effect-detection runs where "
                             "canonical few-shot demonstrations mute the calibration effect.")
    parser.add_argument("--eval_korean_ppl", action="store_true",
                        help="Also compute perplexity on held-out MC4-ko text. "
                             "Continuous, high-sensitivity Korean metric (more sensitive "
                             "to weight-level changes than Korean MCQ tasks).")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    ppl_korean = None
    if args.eval_korean_ppl:
        ppl_korean = eval_ppl_korean(args, model, tokenizer, device)
        print(f"mc4-ko perplexity {ppl_korean}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    ppl_ko_str = f"{ppl_korean:.4f}" if ppl_korean is not None else "-"
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test\tppl_korean", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{ppl_ko_str}", file=f, flush=True)

    downstream_results = None
    nlu_avg = None
    if args.eval_zero_shot:
        task_configs = get_task_configs(args.eval_tasks)
        if args.override_shot is not None:
            task_configs = [(t, args.override_shot) for t, _ in task_configs]
        print(f"running eval task set '{args.eval_tasks}': "
              f"{[(t, n) for t, n in task_configs]}")
        # Convert "auto" string to keyword, integers to int.
        bs = args.eval_batch_size
        if bs != "auto":
            try:
                bs = int(bs)
            except ValueError:
                pass  # leave as string (e.g. "auto:1")
        downstream_results = eval_downstream(model, tokenizer, task_configs,
                                             batch_size=bs, limit=args.eval_limit)
        print("********************************")
        print("downstream evaluation results")
        print(downstream_results)

        nlu_avg = compute_nlu_bundle_avg(downstream_results)
        if nlu_avg is not None:
            print(f"\n7-task NLU bundle average accuracy: {nlu_avg:.4f}")

        # Append to the existing log file alongside perplexity.
        with open(save_filepath, "a") as f:
            print("\ntask\tnum_fewshot\tmetric\tvalue", file=f, flush=True)
            for _, task_results in downstream_results.items():
                for inner_task, metrics in task_results.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"{inner_task}\t-\t{metric}\t{value:.4f}", file=f, flush=True)
            if nlu_avg is not None:
                print(f"nlu_bundle_avg\t-\tacc\t{nlu_avg:.4f}", file=f, flush=True)

    write_manifest(args.save, args, sparsity_ratio, ppl_test, ppl_korean,
                   downstream_results, nlu_avg)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()