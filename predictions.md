# Pre-Registered Predictions — Wanda Stress Test

**Authors:** Jihun Park, Minguhn Kim (POSTECH)
**Pre-registered:** 2026-05-12, by Jihun Park; pending co-author review.
**Status:** Phase 0 not yet run. Phase 1+ not yet started. Numbers below reflect predictions, not measurements.
**Purpose:** Commit to expected effect sizes and directions BEFORE seeing experimental results, as a defense against post-hoc rationalization (garden of forking paths). Amendments after Phase 1 begins must be timestamped and justified in §10.

## 1. Thesis under test

**H1 (proposal-stated):** Wanda's calibration data determines which model capabilities survive pruning. Domain-mismatched calibration can make Wanda worse than calibration-free magnitude pruning on capabilities unrepresented in the calibration data.

**H1' (variance-ratio reformulation, sharper):** Variance in pruning quality across calibration distributions exceeds variance across calibration sample draws of the same distribution:

```
F = Var(metric | calibration source) / Var(metric | seed within source) > 1
```

Wanda's paper claims robustness to sample *count*; we test distribution. These are different claims.

## 2. Qualitative predictions (high confidence)

**Asymmetry.** English calibration damages Korean tasks more than Korean calibration damages English tasks. Korean web text contains substantial English (loanwords, technical terms, mixed-script); English web text contains essentially no Korean. So Korean calibration partially preserves English features; English calibration is blind to Korean-specific features.

Falsifier for asymmetry: Korean-calibrated Wanda damages English tasks more than English-calibrated Wanda damages Korean tasks (direction reversal).

**Sparsity compounding.** Cross-calibration effect grows with sparsity. The 70% cells should show larger calibration-source differences than the 50% cells. Reason: at higher sparsity, more "marginal" weights get pruned, and which marginal weights are kept depends more strongly on the activation distribution.

## 3. Primary predictions — LLaMA-3-8B, 50% unstructured sparsity

Dense baselines (anchored where measured, estimated where not):
- WikiText-2 ppl: **9.06** (Minguhn smoke at nsamples=128, measured)
- MMLU 5-shot: ~63% (LLaMA-3-8B base, public benchmarks; estimate)
- GSM8K 8-shot: ~50% (LLaMA-3-8B base; estimate)
- KoBEST-HS 0-shot: ~50–55% (estimate)
- KMMLU 5-shot: ~40–45% (estimate; LLaMA-3 has limited Korean)
- MC4-ko ppl: ~18–30 (estimate; Korean is harder for the LLaMA-3 tokenizer)

### 3.1 Wanda + C4 (English calibration)

| Metric | Predicted (3 seeds) | Direction vs dense | Confidence |
|---|---|---|---|
| WikiText-2 ppl | 9.5–11 | small ↑ | high |
| MMLU 5-shot | 59–63% | small ↓ | medium |
| GSM8K 8-shot | 38–46% | moderate ↓ | medium |
| **KoBEST-HS 0-shot** | **40–48%** | **moderate–large ↓** | **medium-low** |
| **KMMLU 5-shot** | **32–38%** | **moderate ↓** | **medium-low** |
| **MC4-ko ppl** | **22–40** | **moderate–large ↑** | **medium** |

### 3.2 Wanda + MC4-ko (Korean calibration)

| Metric | Predicted (3 seeds) | Direction vs dense | Direction vs C4-cal |
|---|---|---|---|
| WikiText-2 ppl | 10–12 | small ↑ | slightly worse |
| MMLU 5-shot | 57–62% | small ↓ | slightly worse |
| GSM8K 8-shot | 36–44% | moderate ↓ | comparable |
| **KoBEST-HS 0-shot** | **47–53%** | **small ↓** | **better** |
| **KMMLU 5-shot** | **38–43%** | **small ↓** | **better** |
| **MC4-ko ppl** | **18–28** | **near-flat** | **better** |

### 3.3 Wanda + The Stack (code) and Wanda + random tokens

- **The Stack (English, code domain):** patterns with C4 on Korean tasks (both fail to activate Korean features). Predicted to be close to C4 on all Korean metrics; possibly slightly worse on natural-language English tasks (MMLU, GSM8K) due to domain shift.
- **Random tokens:** expected worst calibration. Predicted to underperform all other Wanda variants on every task. Useful as the "no signal" floor.

Specific numbers omitted (low prior; will be tabulated post-hoc against the C4 and MC4-ko predictions).

### 3.4 Magnitude pruning (calibration-free baseline)

| Metric | Predicted | Reference |
|---|---|---|
| WikiText-2 ppl | 14–25 | Wanda paper: 14.89 (LLaMA-2-7B) |
| KoBEST-HS 0-shot | 35–48% | unknown — Magnitude on KoBEST is unmeasured |
| KMMLU 5-shot | 28–38% | unknown |

### 3.5 Headline proposal-stated comparison

**Predicted:** English-Wanda KoBEST-HS (40–48%) vs Magnitude KoBEST-HS (35–48%).
**Probability of clean headline effect (English-Wanda < Magnitude by ≥2pp on KoBEST-HS): ~50%.**
Hedge reasoning: Magnitude's behavior on Korean tasks is unmeasured. If Magnitude floors out at chance (~25% on 4-way MCQ), the comparison becomes uninformative. KoBEST-HS sample size (~800) further limits detection power for sub-5pp effects.

This is why we prefer the variance-ratio reformulation (§4) as the primary statistical target.

## 4. Variance-ratio predictions — F = Var(source) / Var(seed)

LLaMA-3-8B, 50% sparsity, four calibration sources (C4, MC4-ko, The Stack, random), three seeds each. F computed per task.

| Metric | Predicted F | Interpretation if observed |
|---|---|---|
| WikiText-2 ppl | 1.5–3 | mild calibration sensitivity |
| MMLU 5-shot | 1.2–2.5 | low |
| GSM8K 8-shot | 2–5 | math reasoning is representation-sensitive |
| **KoBEST-HS** | **3–10** | **calibration dominates seed noise** |
| **KMMLU** | **3–10** | **calibration dominates seed noise** |
| **MC4-ko ppl** | **5–15** | **strongest expected signal (continuous metric)** |

**Primary tests for thesis support:** F > 3 on **KMMLU** AND F > 3 on **MC4-ko ppl**.
**Secondary support:** F > 3 on KoBEST-HS (acknowledged-low-power test; primary failure tolerated).

## 5. Phase 0 (mask Jaccard) predictions

LLaMA-3-8B, per-row Jaccard on the actual Wanda score `|W| · sqrt(s)`, C4 vs MC4-ko, 3 seeds per source.

At 50% sparsity:
- **Within-source baseline (mean of per-pair Jaccards):** 0.88–0.94
- **Cross-source mean:** 0.72–0.85
- **Predicted sigma_delta:** 2–5σ

Predicted Phase 0 verdict: **"EFFECT PRESENT"** or **"MARGINAL — PROCEED"**.
Falsifier (would kill the thesis cheaply): `sigma_delta < 1` across all sparsities → mechanistically dead → reframe.
False-positive risk: 128-sample within-source noise larger than expected, making cross-source look elevated when both are actually saturated.

## 6. Phase 1 (anchor reproduction) — must clear before Phase 3

LLaMA-2-7B, C4 calibration, 50% unstructured Wanda. Reproduction against the Wanda paper's LLaMA-2-7B numbers.

| Metric | Predicted | Wanda paper reference | Tolerance |
|---|---|---|---|
| WikiText-2 ppl | 6.3–6.8 | 6.42 (Table 1) | ±0.2 |
| 7-task NLU avg | 60–64% | ~62% (Table 11) | ±2pp |

**Phase 1 gate:** If either metric falls outside tolerance across all three seeds, Phase 3 does NOT start until the discrepancy is explained (library version, dataset revision, eval harness mismatch, or code bug).

## 7. Analysis decisions (locked before runs)

- **Seed count:** 3 seeds per Phase 3 cell; 5 for the headline cells (C4 / MC4-ko on KMMLU and MC4-ko ppl at 50% and 70%).
- **Seed aggregation:** report mean ± std across seeds.
- **Significance threshold:** measured effect exceeds within-source 2σ.
- **Primary Korean test:** **KMMLU** (35K examples → high power).
- **Secondary Korean test:** **KoBEST-HS** (800 examples → low power; explicitly caveated).
- **Highest-sensitivity Korean test:** **MC4-ko perplexity** (continuous, token-level observations; no MCQ argmax floor).
- **Forbidden moves (pre-committed):**
  - No subset-of-layers analysis chosen after seeing results.
  - No subset-of-tasks chosen after seeing results.
  - No discarding "inconvenient" seeds or runs.
  - No retroactive change to which task is "headline."
  - Any analysis not listed here that influences the paper's conclusion must be flagged as exploratory.

## 8. Things we explicitly do NOT predict

- Whether the LLaMA-2-7B within-English-domain-shift control (Phase 4) will be smaller, equal, or larger than the cross-lingual shift on LLaMA-3-8B. Insufficient prior intuition.
- Whether random-token calibration patterns closer to magnitude or to actual-calibration Wanda. Either is possible.
- KMMLU per-subject scores (we predict only overall KMMLU).
- Mask Jaccard at the per-layer-type level (q/k/v vs MLP). Predictions are layer-aggregated.

## 9. Falsification summary

**Thesis falsified** if ANY of:
- F < 1.5 on both KMMLU AND MC4-ko ppl at 50% sparsity.
- Direction reversal: English-cal Wanda outperforms Korean-cal Wanda on Korean tasks at 50% sparsity.
- Phase 0 verdict is "THESIS MECHANICALLY DEAD" (within > 0.99 AND sigma_delta < 0.5).
- Phase 1 anchor reproduction fails repeatedly across seeds and cannot be resolved.

**Thesis supported (never proven)** if MOST of:
- F > 3 on KMMLU at 50% sparsity.
- F > 3 on MC4-ko ppl at 50% sparsity.
- Korean tasks suffer more under English calibration than English tasks suffer under Korean calibration.
- Effect compounds with sparsity (70% > 60% > 50% on F values).

## 10. Amendment log

### 2026-05-12 — pre-existing Phase 1 partial evidence (prior data, not retroactive)

Note for the record: Minguhn ran `python main.py --model meta-llama/Llama-2-7b-hf --prune_method wanda --calib_data c4 --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama2_7b/unstructured/wanda/` on 2026-04-22 (well before this pre-registration was written) and obtained WikiText-2 ppl = **6.461933135986328**. This falls within the §6 predicted tolerance (6.3–6.8) for the LLaMA-2-7B + C4 + 50% Wanda anchor, against the paper's 6.42.

This is **prior evidence**, not part of the formal Phase 1 run:
- It was produced before the predictions were committed, but at a configuration that exactly matches the Phase 1 anchor cell.
- It was a single-seed run with the pre-manifest codebase, so it has no `manifest.json`, no NLU bundle eval, and no Korean perplexity.
- The formal Phase 1 anchor (`configs/phase1_anchor.yaml`, 3 seeds with `--eval_tasks wanda_nlu`) is still required to satisfy the §6 gate, both for the NLU bundle reproduction and for the format the aggregator consumes.

The prior result strengthens our prior that the formal Phase 1 will pass, but does not substitute for it. No predictions are modified by this amendment.

---

*Pre-registration commitment. Any post-hoc deviation from §7 must be explicitly noted in the paper as exploratory analysis.*
