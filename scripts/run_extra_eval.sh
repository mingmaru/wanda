#!/bin/bash
# Extra evaluation to fill remaining gaps in Tables 2-3.
#
# B. SparseGPT 60%/70% × MMLU+KoBEST (2 cells, ~1.5h)
# C. Wanda(MC4-ko) @ 60% × KoBEST (1 cell, ~25min)
# D. Wanda(C4) 60%/70% × MMLU+KoBEST (2 cells, ~30min each if MMLU crashes)
#
# Re-prunes in-place: existing main_exp dirs get rewritten with full metrics.
# `|| true` allows individual cell failure (Wanda MMLU may crash); script
# continues to next cell. Total: ~2.5-3 hours.

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
NSAMPLES="${NSAMPLES:-128}"
SEED="${SEED:-0}"

LOG="out/extra_eval/run.log"
mkdir -p out/extra_eval

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

# ===== B. SparseGPT 60%/70% × MMLU+KoBEST =====
run_cell "sparsegpt_c4_s60_full"  "out/main_exp/sparsegpt_c4_s60/"  sparsegpt c4      0.6  proposal_no_gsm8k
run_cell "sparsegpt_c4_s70_full"  "out/main_exp/sparsegpt_c4_s70/"  sparsegpt c4      0.7  proposal_no_gsm8k

# ===== C. Wanda(MC4-ko) @ 60% × KoBEST =====
run_cell "wanda_mc4_ko_s60"       "out/main_exp/wanda_mc4_ko_s60/"  wanda     mc4_ko  0.6  kobest_only

# ===== D. Wanda(C4) 60%/70% × MMLU+KoBEST (likely to crash on MMLU) =====
run_cell "wanda_c4_s60_full"      "out/main_exp/wanda_c4_s60/"      wanda     c4      0.6  proposal_no_gsm8k
run_cell "wanda_c4_s70_full"      "out/main_exp/wanda_c4_s70/"      wanda     c4      0.7  proposal_no_gsm8k

{
    echo ""
    echo "===== ALL 5 CELLS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} | tee -a "$LOG"
