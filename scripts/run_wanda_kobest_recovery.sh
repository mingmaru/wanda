#!/bin/bash
# Recovery run for the 4 Wanda calibration variants that crashed during MMLU
# in the previous run_exp1_full_eval.sh attempt.
#
# Why this is needed: MMLU's 56K loglikelihood requests triggers CUDA kernel
# failures on Wanda-pruned LLaMA-3-8B. KoBEST has ~2K requests, much smaller
# load, so it should run cleanly. We re-prune from scratch (no saved models)
# and evaluate KoBEST only.
#
# This OVERWRITES the failed log files in out/exp1_full_eval/wanda_*_s50/
# (which previously contained only WT-2 ppl due to the MMLU crash).
#
# Magnitude and SparseGPT cells from the previous run are kept intact — they
# already have MMLU + KoBEST results saved.
#
# Usage:
#   bash scripts/run_wanda_kobest_recovery.sh
#
# Expected runtime: ~2-3 hours sequential on 1 GPU.

set -e

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
OUT_BASE="${OUT_BASE:-out/exp1_full_eval}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

EVAL_FLAGS="--eval_zero_shot --eval_tasks kobest_only"

mkdir -p "$OUT_BASE"
LOG="$OUT_BASE/recovery.log"

{
    echo "===== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "model:     $MODEL"
    echo "out_base:  $OUT_BASE"
    echo "nsamples:  $NSAMPLES"
    echo "seed:      $SEED"
    echo "eval:      kobest_only (KoBEST HellaSwag 0-shot)"
    echo "scope:     4 Wanda calibration variants only (recovery from MMLU crash)"
    echo ""
} | tee "$LOG"

run_cell() {
    local name=$1
    local calib=$2
    local out_dir="$OUT_BASE/$name"

    {
        echo ""
        echo "===== cell: $name ====="
        echo "  method:    wanda"
        echo "  calib:     $calib"
        echo "  sparsity:  0.5"
        echo "  started:   $(date -u +%H:%M:%S)"
    } | tee -a "$LOG"

    python main.py \
        --model "$MODEL" \
        --prune_method wanda \
        --calib_data "$calib" \
        --sparsity_ratio 0.5 \
        --sparsity_type unstructured \
        --nsamples "$NSAMPLES" \
        --seed "$SEED" \
        --save "$out_dir/" \
        $EVAL_FLAGS 2>&1 | tee -a "$LOG"

    echo "  finished:  $(date -u +%H:%M:%S)" | tee -a "$LOG"
}

# ===== 4 Wanda calibration variants (KoBEST only) =====
run_cell "wanda_c4_s50"           c4
run_cell "wanda_mc4_ko_s50"       mc4_ko
run_cell "wanda_the_stack_s50"    the_stack
run_cell "wanda_random_s50"       random

{
    echo ""
    echo "===== ALL 4 WANDA CELLS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
