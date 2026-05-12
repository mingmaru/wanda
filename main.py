import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_downstream

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
    # Phase 3 main experiment: English knowledge, math, two Korean benchmarks.
    # KMMLU is the second Korean benchmark so the headline Korean result
    # does not rest on KoBEST-HellaSwag alone.
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

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        task_configs = get_task_configs(args.eval_tasks)
        print(f"running eval task set '{args.eval_tasks}': "
              f"{[(t, n) for t, n in task_configs]}")
        results = eval_downstream(model, tokenizer, task_configs, limit=args.eval_limit)
        print("********************************")
        print("downstream evaluation results")
        print(results)

        # Append to the existing log file alongside perplexity.
        with open(save_filepath, "a") as f:
            print("\ntask\tnum_fewshot\tmetric\tvalue", file=f, flush=True)
            for _, task_results in results.items():
                for inner_task, metrics in task_results.items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"{inner_task}\t-\t{metric}\t{value:.4f}", file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()