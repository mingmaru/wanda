#!/bin/bash
# Run Wanda MMLU cells using batch_size=1 workaround for the CUDA crash.
# Each cell: full MMLU 5-shot + KoBEST 0-shot.
# Saves in-place (overwrites existing log_wanda.txt for these cells).
#
# Default = Critical 4 cells (~8h):
#   Wanda(C4)     @ 50%, 70%
#   Wanda(MC4-ko) @ 50%, 70%
#
# Set EXTENDED=1 to also run Stack @ 50% and Random @ 50% (+~4h, total ~12h).

set +e   # don't stop on cell failure

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

LOG="out/wanda_mmlu_runs/run.log"
mkdir -p out/wanda_mmlu_runs

{
    echo "===== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "model:        $MODEL"
    echo "batch_size:   1 (CUDA workaround)"
    echo "nsamples:     $NSAMPLES"
    echo "seed:         $SEED"
    echo "extended:     ${EXTENDED:-0}"
    echo ""
} | tee "$LOG"

run_cell() {
    local name=$1
    local out_dir=$2
    local method=$3
    local calib=$4
    local sparsity=$5

    {
        echo ""
        echo "===== cell: $name ====="
        echo "  out_dir:   $out_dir"
        echo "  method:    $method  calib: $calib  sparsity: $sparsity"
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
        --save "$out_dir" \
        --eval_zero_shot --eval_tasks proposal_no_gsm8k \
        --eval_batch_size 1 2>&1 | tee -a "$LOG"

    echo "  finished:  $(date -u +%H:%M:%S)" | tee -a "$LOG"
}

# ===== Critical 4 cells =====
run_cell "wanda_c4_s50_mmlu"      "out/main_exp/wanda_c4_s50/"      wanda c4      0.5
run_cell "wanda_mc4_ko_s50_mmlu"  "out/main_exp/wanda_mc4_ko_s50/"  wanda mc4_ko  0.5
run_cell "wanda_c4_s70_mmlu"      "out/main_exp/wanda_c4_s70/"      wanda c4      0.7
run_cell "wanda_mc4_ko_s70_mmlu"  "out/main_exp/wanda_mc4_ko_s70/"  wanda mc4_ko  0.7

# ===== Optional extended cells (Stack + Random @ 50%) =====
if [[ "${EXTENDED:-0}" == "1" ]]; then
    run_cell "wanda_the_stack_s50_mmlu" "out/main_exp/wanda_the_stack_s50/" wanda the_stack 0.5
    run_cell "wanda_random_s50_mmlu"    "out/main_exp/wanda_random_s50/"    wanda random    0.5
fi

{
    echo ""
    echo "===== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
