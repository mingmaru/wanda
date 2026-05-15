#!/bin/bash
# Extended evaluation: fill remaining gaps in Tables 1-3.
#
# Cells (8 total, ~4-6 hours wall-clock):
#   1. Dense LLaMA-3-8B × {MMLU, GSM8K, KoBEST}      (~1.5-2 hours)
#   2. Wanda + MC4-ko @ 70% × KoBEST                  (~25 min)
#   3-5. Mag/SparseGPT/Wanda + C4 @ 60% × KoBEST      (~1.5 hours)
#   6-8. Mag/SparseGPT/Wanda + C4 @ 70% × KoBEST      (~1.5 hours)
#
# We re-prune in-place into the existing main_exp directories so log files
# (which were WT-2 ppl only) get rewritten with WT-2 ppl + KoBEST.
#
# Dense gets a separate dir (out/extended_eval/dense_s0/) since the existing
# out/dense_llama3/ measured only WT-2 ppl.
#
# Failures are allowed (|| true): if Dense GSM8K crashes with CUDA the
# script continues; we'll have whatever metrics did complete.

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

LOG="out/extended_eval/run.log"
mkdir -p out/extended_eval

{
    echo "===== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "model:     $MODEL"
    echo "nsamples:  $NSAMPLES"
    echo "seed:      $SEED"
    echo ""
} | tee "$LOG"

run_cell() {
    local name=$1
    local out_dir=$2
    local method=$3
    local calib=$4
    local sparsity=$5
    local bundle=$6

    {
        echo ""
        echo "===== cell: $name ====="
        echo "  out_dir:   $out_dir"
        echo "  method:    $method"
        echo "  calib:     $calib"
        echo "  sparsity:  $sparsity"
        echo "  bundle:    $bundle"
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
        --eval_zero_shot --eval_tasks "$bundle" 2>&1 | tee -a "$LOG" || true

    echo "  finished:  $(date -u +%H:%M:%S)" | tee -a "$LOG"
}

# ===== 1. Dense LLaMA-3-8B × full proposal eval =====
# Try MMLU + GSM8K + KoBEST. If GSM8K crashes, we still get MMLU + KoBEST
# (per the manifest order in main.py).
run_cell "dense_s0"              "out/extended_eval/dense_s0/"          wanda     c4      0.0  proposal

# ===== 2. Exp 3: Wanda + MC4-ko @ 70% × KoBEST =====
run_cell "wanda_mc4_ko_s70"      "out/main_exp/wanda_mc4_ko_s70/"       wanda     mc4_ko  0.7  kobest_only

# ===== 3-5. Sparsity sweep @ 60% × KoBEST =====
run_cell "magnitude_s60"         "out/main_exp/magnitude_s60/"          magnitude c4      0.6  kobest_only
run_cell "sparsegpt_c4_s60"      "out/main_exp/sparsegpt_c4_s60/"       sparsegpt c4      0.6  kobest_only
run_cell "wanda_c4_s60"          "out/main_exp/wanda_c4_s60/"           wanda     c4      0.6  kobest_only

# ===== 6-8. Sparsity sweep @ 70% × KoBEST =====
run_cell "magnitude_s70"         "out/main_exp/magnitude_s70/"          magnitude c4      0.7  kobest_only
run_cell "sparsegpt_c4_s70"      "out/main_exp/sparsegpt_c4_s70/"       sparsegpt c4      0.7  kobest_only
run_cell "wanda_c4_s70"          "out/main_exp/wanda_c4_s70/"           wanda     c4      0.7  kobest_only

{
    echo ""
    echo "===== ALL 8 CELLS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
