# Wanda Stress Test (POSTECH Efficient ML Final Project)

This fork extends [locuslab/wanda](https://github.com/locuslab/wanda) for a stress-testing study of Wanda's robustness to calibration-data distribution. The thesis: Wanda's published robustness claim is about calibration sample *count*; it does not generalize to calibration *distribution*. Domain-mismatched calibration (especially cross-lingual: English → Korean) can produce systematically different pruning masks and downstream eval degradation.

**Read first:** [`predictions.md`](predictions.md) is the pre-registered prediction document. All thresholds, seed counts, headline tasks, and forbidden moves are committed there before Phase 3 runs land. Any analysis decision not in §7 of `predictions.md` is exploratory.

## Phases (from `predictions.md` §3-6)

| Phase | What it does | Code | Config |
|---|---|---|---|
| 0 | Pre-experiment: per-row mask Jaccard under dense propagation (kill-switch) | `phase0.py` | run directly |
| 1 | LLaMA-2-7B anchor reproduction (Wanda paper Table 11 numbers) | `main.py` | `configs/phase1_anchor.yaml` |
| 3 | Main: 4 calibrations × {50%, 70%} × {5 seeds headline, 3 seeds secondary} | `main.py` | `configs/phase3_main.yaml` |
| 4 | LLaMA-2-7B within-English domain-shift control | `main.py` | `configs/phase4_within_en.yaml` |
| 5 | Magnitude + SparseGPT reference baselines | `main.py` | `configs/phase5_baselines.yaml` |
| 6 | Mask Jaccard analysis (mechanism) | `analyze_masks.py` | post-hoc on Phase 3+5 |

## Workflow

```bash
# 1. Phase 0 cheap kill-switch (~1 GPU-day on LLaMA-3-8B)
python phase0.py --model meta-llama/Meta-Llama-3-8B \
    --sources c4 mc4_ko --seeds 0 1 2 --sparsities 0.5 0.6 0.7 \
    --critical_pair c4-mc4_ko --save out/phase0_llama3/

# 2. Phase 1 anchor (~3 hours sequential, ~1 hour parallel)
python orchestrator.py --config configs/phase1_anchor.yaml --max_parallel 3 \
    --gpu_ids 0,1,2

# 3. Phase 3 main (~32h sequential, ~4-5h parallel on 8x A6000)
python orchestrator.py --config configs/phase3_main.yaml --max_parallel 8 \
    --gpu_ids 0,1,2,3,4,5,6,7

# 4. Phase 4 within-English control (~6h sequential, ~2h parallel)
python orchestrator.py --config configs/phase4_within_en.yaml --max_parallel 3

# 5. Phase 5 reference baselines
python orchestrator.py --config configs/phase5_baselines.yaml --max_parallel 8

# 6. Aggregate Phase 3 results and compute F-ratios
python aggregate.py --runs_dir out/phase3/ --out out/aggregated/

# 7. Generate report tables
python fill_tables.py --summary out/aggregated/summary.csv \
    --out final_report/tables/

# 8. Phase 6 mask Jaccard analysis
python analyze_masks.py \
    --runs_by_source c4=out/phase3/wanda_c4_50_seed* mc4_ko=out/phase3/wanda_mc4ko_50_seed* \
    --method wanda --out out/phase6_c4_vs_mc4ko_50/

# 9. (Optional) Phase 0 vs Phase 6 alignment check
python analyze_phase0_vs_phase6.py \
    --phase0_stats out/phase0_llama3/phase0_stats.json \
    --phase6_csv   out/phase6_c4_vs_mc4ko_50/per_layer_jaccard.csv \
    --pair c4-mc4_ko --sparsity 0.50 \
    --out out/alignment_c4_vs_mc4ko_50/
```

## New CLI flags on `main.py`

| Flag | Default | Purpose |
|---|---|---|
| `--calib_data` | `c4` | Calibration source: `c4`, `mc4_ko`, `the_stack`, `random` |
| `--eval_zero_shot` | off | Run downstream eval (bundle selected by `--eval_tasks`) |
| `--eval_tasks` | `headline` | `headline` (Phase 3), `wanda_nlu` (Phase 1 anchor), `all` |
| `--eval_korean_ppl` | off | Compute perplexity on held-out MC4-ko (continuous Korean metric) |
| `--eval_limit` | None | Limit examples per task (debugging) |
| `--override_shot` | None | Override num_fewshot for every task (e.g., 0 for higher-power Phase 3) |

## Per-run output

Every `main.py` run writes to `args.save/`:

| File | Content |
|---|---|
| `log_{method}.txt` | TSV: actual sparsity, WikiText-2 ppl, MC4-ko ppl, per-task eval results |
| `masks_{method}.pt` | bool tensor per layer; Phase 6's input |
| `manifest.json` | full reproducibility record (args + library versions + results) |
| `orchestrator.log` | only when launched via `orchestrator.py` |

## Orchestrator config schema

```yaml
name: my_runs
defaults:
  # Any main.py CLI arg
  model: meta-llama/Meta-Llama-3-8B
  sparsity_type: unstructured
  nsamples: 128
  eval_zero_shot: true
  # Subprocess environment overrides (merged into os.environ)
  env:
    CUDA_VISIBLE_DEVICES: "0"

runs:
  - prune_method: wanda
    calib_data: c4
    sparsity_ratio: 0.5
    seeds: [0, 1, 2]                         # expands into one cell per seed
    save_template: out/x/seed{seed}/         # {seed} and other scalars are interpolated
```

Booleans True become bare flags; False is omitted. The `env` key is merged into the subprocess environment instead of being emitted as a CLI flag. `--max_parallel N` runs N cells concurrently with auto-assigned `CUDA_VISIBLE_DEVICES` from `--gpu_ids` (default `0..N-1`).

## Dependencies

| Package | Used by | Notes |
|---|---|---|
| `torch` | everything | Pinned via standard requirements |
| `transformers` | `main.py`, `phase0.py` | 4.45.2 verified (see Minguhn's `.claude/settings.json`) |
| `datasets` | calibration loaders | |
| `lm-eval` >= 0.4 | `lib/eval.py` | `simple_evaluate` + `HFLM` API |
| `pyyaml` | `orchestrator.py` | Usually pre-installed via HF stack |
| `scipy` | not required | |

## Cell-level run convention

Save paths in our configs use `out/<phase>/<method>_<calib>_<sparsity>_seed{seed}/`. This is phase-organized so `aggregate.py` can walk by phase. Minguhn's earlier convention `out/<model>/<sparsity_type>/<method>/` is fully overridable per-cell via `save_template`.

## What this fork adds on top of upstream Wanda

- 3 new calibration loaders (`mc4_ko`, `the_stack`, `random`) in `lib/data.py`
- Generalized `layer_kwargs` plumbing in `lib/prune.py` (works with current `transformers` LLaMA-3 forward signature)
- lm-eval-harness v0.4+ migration in `lib/eval.py`
- KMMLU, KoBEST-HellaSwag, and MC4-ko perplexity for Korean evaluation
- Pruning mask saving in all three pruners (`prune_wanda`, `prune_magnitude`, `prune_sparsegpt`)
- `phase0.py` pre-experiment with corrected per-row Jaccard metric + sigma-based verdict
- `orchestrator.py` YAML-driven runner with bounded parallelism
- `aggregate.py` + F-ratio (pre-registered statistical test from `predictions.md` §4)
- `fill_tables.py` LaTeX table generators
- `analyze_masks.py` Phase 6 mechanism analyzer
- `analyze_phase0_vs_phase6.py` methodological cross-check
- `tests/test_phase0_math.py` synthetic verification (12 tests; runs in ~1 second)

The upstream Wanda paper, citation, and original documentation follow below.

---

# Pruning LLMs by Weights and Activations
Official PyTorch implementation of **Wanda** (Pruning by **W**eights **and a**ctivations), as presented in our paper:

**A Simple and Effective Pruning Approach for Large Language Models** </br>
*Mingjie Sun\*, Zhuang Liu\*, Anna Bair, J. Zico Kolter* (* indicates equal contribution) <br>
Carnegie Mellon University, Meta AI Research and Bosch Center for AI  <br>
[Paper](https://arxiv.org/abs/2306.11695) - [Project page](https://eric-mingjie.github.io/wanda/home.html)

```bibtex
@article{sun2023wanda,
  title={A Simple and Effective Pruning Approach for Large Language Models}, 
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J. Zico},
  year={2023},
  journal={arXiv preprint arXiv:2306.11695}
}
```

--- 
<p align="center">
<img src="https://user-images.githubusercontent.com/20168304/273351964-53c3807e-3453-49c5-b855-b620b1026466.png" width=100% height=100% 
class="center">
</p>

Compared to magnitude pruning which removes weights solely based on their magnitudes, our pruning approach **Wanda** removes weights on a *per-output* basis, by the product of weight magnitudes and input activation norms.

## Update
- [x] (9.22.2023) Add [support](https://github.com/locuslab/wanda#pruning-llama-2) for LLaMA-2.
- [x] (9.22.2023) Add [code](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) to reproduce the ablation study on OBS weight update in the paper.
- [x] (10.6.2023) Add new [support](https://github.com/locuslab/wanda#ablation-on-obs-weight-update) for the weight update analysis in the ablation study. Feel free to try it out!
- [x] (10.6.2023) Add [support](https://github.com/locuslab/wanda#zero-shot-evaluation) for zero-shot evaluation.
- [x] (10.20.2023) Add code for pruning OPT models.
- [x] (10.23.2023) Add code for [LoRA fine-tuning](lora_ft).

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
The [scripts](scripts) directory contains all the bash commands to replicate the main results (Table 2) in our paper.

Below is an example command for pruning LLaMA-7B with Wanda, to achieve unstructured 50% sparsity.
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```sh
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/ 
```

### Pruning LLaMA-2
For [LLaMA-2](https://ai.meta.com/llama/) models, replace `--model` with `meta-llama/Llama-2-7b-hf` (take `7b` as an example):
```sh 
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```
LLaMA-2 results: (LLaMA-2-34b is not released as of 9.22.2023)
|sparsity| ppl              | llama2-7b | llama2-13b | llama2-70b |
|------|------------------|----------|------------|------------|
|-| dense            | 5.12     | 4.57       | 3.12     |
|unstructured 50%| magnitude        | 14.89    | 6.37       | 4.98     |
|unstructured 50%| sparsegpt        | 6.51     | 5.63       | **3.98**  |
|unstructured 50%| wanda            | **6.42** | **5.56**   | **3.98**  |
|4:8| magnitude        | 16.48    | 6.76       | 5.58     |
|4:8| sparsegpt        | 8.12     | 6.60      | 4.59     |
|4:8| wanda            | **7.97** | **6.55**  | **4.47**     |
|2:4| magnitude        | 54.59    | 8.33       | 6.33       |
|2:4| sparsegpt        | **10.17** | 8.32       | 5.40      |
|2:4| wanda            | 11.02    | **8.27**   | **5.16**     |

### Ablation on OBS weight update
To reproduce the analysis on weight update, we provide our implementation for this ablation. All commands can be found in [this script](scripts/ablate_weight_update.sh).
```sh
for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model decapoda-research/llama-7b-hf \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method ${method} \
  --save out/llama_7b_ablation/unstructured/
done 
```
Here `ablate_{mag/wanda}_{seq/iter}` means that we use magnitude pruning or wanda to obtain the pruned mask at each layer, then apply weight update procedure with either a sequential style or an iterative style every 128 input channels. For details, please see Section 5 of our [paper](https://arxiv.org/abs/2306.11695).

### Zero-Shot Evaluation
For evaluating zero-shot tasks, we modify the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework so that it could evaluate pruned LLM models. We provide the modified repo in [this link](https://drive.google.com/file/d/1zugbLyGZKsH1L19L9biHLfaGGFnEc7XL/view?usp=sharing). Make sure to download, extract and install this custom `lm_eval` package from the source code.

For reproducibility, we used [commit `df3da98`](https://github.com/EleutherAI/lm-evaluation-harness/tree/df3da98c5405deafd519c2ddca52bb7c3fe36bef) on the main branch. All tasks were evaluated on task version of 0 except for BoolQ, where the task version is 1.

On a high level, the functionality we provide is adding two arguments `pretrained_model` and `tokenizer` in this [function](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/evaluator.py#L17). We can then call this `simple_evaluate` function API from our [codebase](https://github.com/locuslab/wanda/blob/main/lib/eval.py#L148) to evaluate sparse pruned LLMs. To evaluate zero-shot tasks in addition to the WikiText perplexity, pass the `--eval_zero_shot` argument. 

### Speedup Evaluation
The pruning speed for each method is evaluated by the cumulated time spent on pruning (for each layer), without the forward passes.

For inference speedup with structured sparsity, we refer the reader to this [blog post](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html), where  structured sparsity is supported by `PyTorch >= 2.1`. You can switch between the CUTLASS or CuSPARSELt kernel [here](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/sparse/semi_structured.py#L55).

Last, for pruning image classifiers, see directory [image_classifiers](image_classifiers) for details.

## Acknowledgement
This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Questions
Feel free to discuss papers/code with us through issues/emails!

mingjies at cs.cmu.edu  
liuzhuangthu at gmail.com 