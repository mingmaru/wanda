#!/bin/bash
# Main experiment driver: Exp 1 (6) + Exp 2 (6 new) + Exp 3 (1 new) = 13 cells on LLaMA-3-8B.
#
# Usage:
#   bash scripts/run_main_experiments.sh                # perplexity-only first pass
#   FULL_EVAL=1 bash scripts/run_main_experiments.sh    # add MMLU/GSM8K/KoBEST/KMMLU eval
#
# Environment overrides (optional):
#   MODEL=...        # default meta-llama/Meta-Llama-3-8B
#   OUT_BASE=...     # default out/main_exp
#   NSAMPLES=...     # default 128
#   SEED=...         # default 0
#
# Runs sequentially on a single GPU. Each cell:
#   - prunes the model with given method/calibration/sparsity
#   - measures WikiText-2 perplexity
#   - (if FULL_EVAL=1) measures MMLU/GSM8K/KoBEST-HS/KMMLU via lm-eval-harness
#   - writes manifest.json + log_{method}.txt under $OUT_BASE/<cell_name>/

set -e

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
OUT_BASE="${OUT_BASE:-out/main_exp}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

# Default first pass: WikiText-2 perplexity (proposal's primary metric, always runs).
# Strict proposal scope: no MC4-ko ppl, no KMMLU.
EVAL_FLAGS=""
EVAL_NOTE="perplexity only (WikiText-2)"
if [[ "${FULL_EVAL:-0}" == "1" ]]; then
    # Proposal's 4-metric eval set: WT2 ppl (already on) + MMLU + GSM8K + KoBEST-HS.
    EVAL_FLAGS="--eval_zero_shot --eval_tasks proposal"
    EVAL_NOTE="proposal eval (WikiText-2 ppl + MMLU + GSM8K + KoBEST-HS)"
fi

mkdir -p "$OUT_BASE"
LOG="$OUT_BASE/run.log"

{
    echo "===== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "model:     $MODEL"
    echo "out_base:  $OUT_BASE"
    echo "nsamples:  $NSAMPLES"
    echo "seed:      $SEED"
    echo "eval:      $EVAL_NOTE"
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

# ===== Exp 2: sparsity sweep 60% (3 cells) =====
run_cell "magnitude_s60"          magnitude  c4         0.6
run_cell "sparsegpt_c4_s60"       sparsegpt  c4         0.6
run_cell "wanda_c4_s60"           wanda      c4         0.6

# ===== Exp 2: sparsity sweep 70% (3 cells) =====
run_cell "magnitude_s70"          magnitude  c4         0.7
run_cell "sparsegpt_c4_s70"       sparsegpt  c4         0.7
run_cell "wanda_c4_s70"           wanda      c4         0.7

# ===== Exp 3: calibration x sparsity corner (1 cell) =====
run_cell "wanda_mc4_ko_s70"       wanda      mc4_ko     0.7

{
    echo ""
    echo "===== ALL 13 CELLS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
