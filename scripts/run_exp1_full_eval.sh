#!/bin/bash
# Exp 1 (50% sparsity) full evaluation — 6 cells with proposal's 4-metric eval.
#
# Re-prunes from scratch (pruned models from first pass were not saved) and
# runs 3 proposal metrics: WikiText-2 ppl + MMLU (5-shot) + KoBEST HellaSwag
# (0-shot). GSM8K (8-shot CoT) is excluded — it caused CUDA kernel failures
# on pruned LLaMA-3-8B during generation in the previous attempt; documented
# in the final report.
#
# Usage:
#   bash scripts/run_exp1_full_eval.sh
#
# Environment overrides:
#   MODEL=...        # default meta-llama/Meta-Llama-3-8B
#   OUT_BASE=...     # default out/exp1_full_eval
#   NSAMPLES=...     # default 128
#   SEED=...         # default 0
#
# Expected runtime: ~3-5 hours sequential on 1 GPU.

set -e

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
OUT_BASE="${OUT_BASE:-out/exp1_full_eval}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

EVAL_FLAGS="--eval_zero_shot --eval_tasks proposal_no_gsm8k"

mkdir -p "$OUT_BASE"
LOG="$OUT_BASE/run.log"

{
    echo "===== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "model:     $MODEL"
    echo "out_base:  $OUT_BASE"
    echo "nsamples:  $NSAMPLES"
    echo "seed:      $SEED"
    echo "eval:      proposal bundle (WT2 ppl + MMLU 5-shot + GSM8K 8-shot + KoBEST-HS 0-shot)"
    echo ""
} | tee "$LOG"

run_cell() {
    local name=$1
    local method=$2
    local calib=$3
    local sparsity=$4
    local out_dir="$OUT_BASE/$name"

    {
        echo ""
        echo "===== cell: $name ====="
        echo "  method:    $method"
        echo "  calib:     $calib"
        echo "  sparsity:  $sparsity"
        echo "  started:   $(date -u +%H:%M:%S)"
    } | tee -a "$LOG"

    python main.py \
        --model "$MODEL" \
        --prune_method "$method" \
        --calib_data "$calib" \
        --sparsity_ratio "$sparsity" \
        --sparsity_type unstructured \
        --nsamples "$NSAMPLES" \
        --seed "$SEED" \
        --save "$out_dir/" \
        $EVAL_FLAGS 2>&1 | tee -a "$LOG"

    echo "  finished:  $(date -u +%H:%M:%S)" | tee -a "$LOG"
}

# ===== Exp 1: 50% sparsity, 4 calibrations + 2 baselines (6 cells) =====
run_cell "magnitude_s50"          magnitude  c4         0.5
run_cell "sparsegpt_c4_s50"       sparsegpt  c4         0.5
run_cell "wanda_c4_s50"           wanda      c4         0.5
run_cell "wanda_mc4_ko_s50"       wanda      mc4_ko     0.5
run_cell "wanda_the_stack_s50"    wanda      the_stack  0.5
run_cell "wanda_random_s50"       wanda      random     0.5

{
    echo ""
    echo "===== ALL 6 CELLS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
