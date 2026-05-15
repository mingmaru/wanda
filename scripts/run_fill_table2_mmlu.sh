#!/bin/bash
# Fill the remaining MMLU cells in Table 2:
#   - Magnitude  @ 60%, 70%   (no batch_size=1 needed, ~45 min each)
#   - Wanda(C4)  @ 60%        (batch_size=1, ~1.5-2h)
#   - Wanda(MC4-ko) @ 60%     (batch_size=1, ~1.5-2h)
# Total: ~4.5-5.5 hours.
#
# Magnitude is run with batch_size=auto (no CUDA issue, faster).
# Wanda 60% cells use batch_size=1 (CUDA workaround) and the existing
# log_wanda.txt + manifest.json will be overwritten — backups created first.

set +e

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

LOG="out/fill_table2/run.log"
mkdir -p out/fill_table2

{
    echo "===== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "model:        $MODEL"
    echo "nsamples:     $NSAMPLES"
    echo "seed:         $SEED"
    echo ""
} | tee "$LOG"

# ===== Backup Wanda 60% cells before overwriting =====
echo "=== Creating backups ===" | tee -a "$LOG"
for c in wanda_c4_s60 wanda_mc4_ko_s60; do
    if [ -f "out/main_exp/$c/log_wanda.txt" ] && [ ! -f "out/main_exp/$c/log_wanda.txt.pre_60mmlu_backup" ]; then
        cp out/main_exp/$c/log_wanda.txt out/main_exp/$c/log_wanda.txt.pre_60mmlu_backup
        cp out/main_exp/$c/manifest.json out/main_exp/$c/manifest.json.pre_60mmlu_backup 2>/dev/null
        echo "  backup: $c" | tee -a "$LOG"
    fi
done

run_cell() {
    local name=$1
    local out_dir=$2
    local method=$3
    local calib=$4
    local sparsity=$5
    local batch=$6

    {
        echo ""
        echo "===== cell: $name ====="
        echo "  out_dir:    $out_dir"
        echo "  method:     $method  calib: $calib  sparsity: $sparsity"
        echo "  batch_size: $batch"
        echo "  started:    $(date -u +%H:%M:%S)"
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
        --eval_batch_size "$batch" 2>&1 | tee -a "$LOG"

    echo "  finished:   $(date -u +%H:%M:%S)" | tee -a "$LOG"
}

# ===== Magnitude 60%, 70% (quick, no batch issue) =====
run_cell "magnitude_s60_mmlu"     "out/main_exp/magnitude_s60/"     magnitude c4     0.6  auto
run_cell "magnitude_s70_mmlu"     "out/main_exp/magnitude_s70/"     magnitude c4     0.7  auto

# ===== Wanda 60% × 2 calibrations (batch_size=1) =====
run_cell "wanda_c4_s60_mmlu"      "out/main_exp/wanda_c4_s60/"      wanda     c4     0.6  1
run_cell "wanda_mc4_ko_s60_mmlu"  "out/main_exp/wanda_mc4_ko_s60/"  wanda     mc4_ko 0.6  1

{
    echo ""
    echo "===== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
