"""
Phase 0 (corrected): pre-experiment for Wanda stress test.

Computes per-row pruning-mask Jaccard on |W[i,j]| * sqrt(scaler_row[j]) -
the exact operation Wanda performs - across calibration sources and seeds.
Within-source seed pairs establish the 128-sample calibration noise floor;
cross-source comparisons are reported as deltas relative to that floor.

Why per-row Jaccard, not Spearman of scaler_row:
    Wanda's score is per-(output, input) entry: |W[i,j]| * sqrt(s[j]).
    Pruning is per output row. Rank-correlating scaler_row across
    calibrations ignores |W|: counterexamples exist where Spearman = 1.0
    but per-row pruning masks completely disagree. Per-row Jaccard on the
    actual score directly measures pruning-decision overlap.

Why within-source seeds:
    A fixed threshold on cross-source Jaccard (e.g. > 0.95 = "dead") is
    unanchored. The relevant question is whether cross-source disagreement
    exceeds within-source noise from sampling 128 random calibration
    windows. Both signals must be measured.

Intrinsic limitation:
    Wanda's pruning is sequential - layer i+1's scaler is computed on
    layer i's pruned outputs. This script captures scalers under fully
    dense propagation, so its mask predictions are most accurate for
    early layers and degrade for later ones. The verdict reports the
    early-layer signal (layers 00-07) separately.

Output:
    phase0_scalers_{source}_seed{N}.pkl  -- raw scaler_row tensors
    phase0_stats.json                    -- per-layer per-sparsity per-pair stats
    phase0_summary.json                  -- aggregated + verdict

Usage:
    python phase0.py --model meta-llama/Meta-Llama-3-8B \
        --sources c4 mc4_ko \
        --seeds 0 1 2 \
        --sparsities 0.5 0.6 0.7 \
        --critical_pair c4-mc4_ko \
        --save out/phase0_llama3/
"""
import argparse
import json
import os
import pickle
from collections import defaultdict
from importlib.metadata import version
from itertools import combinations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.data import get_loaders
from lib.layerwrapper import WrappedGPT
from lib.prune import find_layers, prepare_calibration_input, _move_kwargs_to_device


# ============================================================
# Stage 1: capture scaler_row for one (source, seed) pair
# ============================================================

def capture_scalers(model, tokenizer, calib_source, nsamples, seed, device):
    """Forward-pass calibration data, capture WrappedGPT.scaler_row per Linear.

    Mirrors prune_wanda's forward loop exactly but never modifies weights.
    Returns dict of "layerNN.proj_name" -> 1D CPU tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"  loading calibration: {calib_source} seed={seed} nsamples={nsamples}")
    dataloader, _ = get_loaders(
        calib_source, nsamples=nsamples, seed=seed,
        seqlen=model.seqlen, tokenizer=tokenizer,
    )
    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    scalers = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            layer_kwargs = _move_kwargs_to_device(layer_kwargs, dev)
        wrapped = {name: WrappedGPT(subset[name]) for name in subset}

        def make_hook(name):
            def hook(_, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return hook

        handles = [subset[name].register_forward_hook(make_hook(name)) for name in wrapped]
        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in subset:
            scalers[f"layer{i:02d}.{name}"] = wrapped[name].scaler_row.detach().cpu().clone()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return scalers


# ============================================================
# Stage 2: per-layer mask analysis
# ============================================================

def compute_mask(W_abs, scaler_row, sparsity):
    """Wanda's actual per-row pruning mask.

    Score: |W[i,j]| * sqrt(scaler_row[j]). For each row i, the bottom-K
    indices (K = int(d_in * sparsity)) are True (would be pruned). Uses
    stable sort to match prune.py's behavior on ties.
    """
    d_in = W_abs.shape[1]
    K = int(d_in * sparsity)
    v = W_abs * torch.sqrt(scaler_row).unsqueeze(0)
    _, indices = v.sort(dim=1, stable=True)
    mask = torch.zeros_like(v, dtype=torch.bool)
    mask.scatter_(1, indices[:, :K], True)
    return mask


def per_row_jaccard(mask_a, mask_b):
    """Per-row Jaccard between two bool masks of equal shape.

    Both masks select exactly K dims per row, so |union| = 2K - |inter|
    and jaccard = inter / (2K - inter). Returns 1D tensor of length d_out.
    """
    inter = (mask_a & mask_b).sum(dim=1).float()
    union = (mask_a | mask_b).sum(dim=1).float()
    return torch.where(union > 0, inter / union, torch.ones_like(inter))


def jaccard_stats(arr):
    """Aggregate a 1D numpy array of per-row Jaccards."""
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n_jaccards": int(len(arr)),
    }


def analyze_layer(linear, layer_name, captures, sparsities):
    """For one Linear layer, compute Jaccard stats at each sparsity for
    every pair of (source, seed) captures.

    Within each source-class (and cross pair), reports the standard
    aggregated jaccard_stats over all rows x pairs PLUS the per-pair means
    and their std. The within-source pair_mean_std is the noise floor used
    in the sigma-based verdict.
    """
    W_abs = linear.weight.data.abs().float().cpu()
    d_out, d_in = W_abs.shape

    results = {"n_rows": d_out, "n_cols": d_in, "params": d_out * d_in}
    entities = list(captures.keys())

    for sparsity in sparsities:
        spk = f"{sparsity:.2f}"
        masks = {}
        for (src, sd), scaler_dict in captures.items():
            s = scaler_dict[layer_name].float().cpu()
            masks[(src, sd)] = compute_mask(W_abs, s, sparsity)

        within_rows = defaultdict(list)
        cross_rows = defaultdict(list)
        within_pair_means = defaultdict(list)
        cross_pair_means = defaultdict(list)

        for e1, e2 in combinations(entities, 2):
            j = per_row_jaccard(masks[e1], masks[e2]).numpy()
            pair_mean = float(np.mean(j))
            src1, _ = e1
            src2, _ = e2
            if src1 == src2:
                within_rows[src1].append(j)
                within_pair_means[src1].append(pair_mean)
            else:
                key = "-".join(sorted([src1, src2]))
                cross_rows[key].append(j)
                cross_pair_means[key].append(pair_mean)

        def with_pair_stats(rows, pair_means):
            stats = jaccard_stats(np.concatenate(rows))
            stats["pair_means"] = [float(x) for x in pair_means]
            stats["n_pairs"] = len(pair_means)
            stats["pair_mean_std"] = (
                float(np.std(pair_means, ddof=1)) if len(pair_means) >= 2 else None
            )
            return stats

        results[spk] = {
            "within": {src: with_pair_stats(within_rows[src], within_pair_means[src])
                       for src in within_rows},
            "cross": {key: with_pair_stats(cross_rows[key], cross_pair_means[key])
                      for key in cross_rows},
        }
        del masks

    return results


# ============================================================
# Stage 3: aggregate across layers
# ============================================================

def aggregate_across_layers(per_layer_stats, sparsities, weighting):
    """Aggregate per-layer means into model-level numbers.

    weighting="param" weights each layer by d_out * d_in (matches Wanda's
    actual operation scope). "uniform" weights each layer equally.

    pair_mean_std is aggregated where defined (None for layers/pairs with
    <2 pairs); the running mean uses only contributions where the value
    is present, so a single missing layer does not zero out the aggregate.
    """
    metrics = ["mean", "p10", "p50", "p90", "pair_mean_std"]
    aggregated = {}
    for sparsity in sparsities:
        spk = f"{sparsity:.2f}"
        within_keys = set()
        cross_keys = set()
        for layer_stats in per_layer_stats.values():
            within_keys.update(layer_stats[spk]["within"].keys())
            cross_keys.update(layer_stats[spk]["cross"].keys())

        within_agg = {k: {m: 0.0 for m in metrics} for k in within_keys}
        cross_agg = {k: {m: 0.0 for m in metrics} for k in cross_keys}
        within_w = {k: {m: 0.0 for m in metrics} for k in within_keys}
        cross_w = {k: {m: 0.0 for m in metrics} for k in cross_keys}

        for layer_stats in per_layer_stats.values():
            w = layer_stats["params"] if weighting == "param" else 1.0
            for k, st in layer_stats[spk]["within"].items():
                for m in metrics:
                    v = st.get(m)
                    if v is not None:
                        within_agg[k][m] += v * w
                        within_w[k][m] += w
            for k, st in layer_stats[spk]["cross"].items():
                for m in metrics:
                    v = st.get(m)
                    if v is not None:
                        cross_agg[k][m] += v * w
                        cross_w[k][m] += w

        for k in within_agg:
            for m in metrics:
                within_agg[k][m] = within_agg[k][m] / within_w[k][m] if within_w[k][m] > 0 else None
        for k in cross_agg:
            for m in metrics:
                cross_agg[k][m] = cross_agg[k][m] / cross_w[k][m] if cross_w[k][m] > 0 else None

        aggregated[spk] = {"within": within_agg, "cross": cross_agg}
    return aggregated


# ============================================================
# Stage 4: verdict
# ============================================================

EARLY_LAYER_PREFIXES = tuple(f"layer{i:02d}" for i in range(8))


def verdict(per_layer_stats, aggregated, critical_pair, sources, sparsities):
    """Render headline verdict for the critical cross-source pair.

    Reports delta in both absolute Jaccard units and sigma units, where
    sigma = std across per-pair within-source means. Sigma units make
    the verdict invariant to whether 128 samples saturates within-source
    estimation (high mean, tiny std) or leaves residual noise (lower
    mean, larger std) - the ratio of cross-effect to within-noise is
    the scale-invariant quantity.

    When pair_mean_std is unavailable (only 1 seed per source), falls
    back to absolute thresholds with a warning.
    """
    print("\n" + "=" * 70)
    print("PHASE 0 VERDICT")
    print("=" * 70)

    src_a, src_b = sorted(critical_pair.split("-"))
    critical_key = f"{src_a}-{src_b}"
    if src_a not in sources or src_b not in sources:
        print(f"  critical pair '{critical_pair}' not found in --sources; skipping.")
        return {}

    summary = {}
    for sparsity in sparsities:
        spk = f"{sparsity:.2f}"
        wa = aggregated[spk]["within"].get(src_a, {})
        wb = aggregated[spk]["within"].get(src_b, {})
        cab = aggregated[spk]["cross"].get(critical_key, {})
        within_a, within_b, cross_ab = wa.get("mean"), wb.get("mean"), cab.get("mean")
        within_a_std, within_b_std = wa.get("pair_mean_std"), wb.get("pair_mean_std")

        if None in (within_a, within_b, cross_ab):
            print(f"  sparsity {sparsity}: missing data (need >=2 seeds per source).")
            continue

        baseline = min(within_a, within_b)
        delta = baseline - cross_ab
        # Tighter source = smaller std = more conservative sigma test
        within_stds = [s for s in (within_a_std, within_b_std) if s is not None and s > 0]
        sigma = min(within_stds) if within_stds else None
        sigma_delta = (delta / sigma) if sigma else None

        # Worst-layer drill-down (per-layer sigma uses that layer's own std)
        worst_layer, worst_delta, worst_sigma_delta = None, -1e9, None
        for ln, ls in per_layer_stats.items():
            la = ls[spk]["within"].get(src_a, {})
            lb = ls[spk]["within"].get(src_b, {})
            lc = ls[spk]["cross"].get(critical_key, {})
            w_a, w_b, c_ab = la.get("mean"), lb.get("mean"), lc.get("mean")
            if None in (w_a, w_b, c_ab):
                continue
            d = min(w_a, w_b) - c_ab
            if d > worst_delta:
                worst_delta, worst_layer = d, ln
                lws = [s for s in (la.get("pair_mean_std"), lb.get("pair_mean_std"))
                       if s is not None and s > 0]
                worst_sigma_delta = (d / min(lws)) if lws else None

        # Early-layer aggregate (Phase 0's reliable regime)
        early_w, early_c, early_s = [], [], []
        for ln, ls in per_layer_stats.items():
            if not ln.startswith(EARLY_LAYER_PREFIXES):
                continue
            la = ls[spk]["within"].get(src_a, {})
            lb = ls[spk]["within"].get(src_b, {})
            lc = ls[spk]["cross"].get(critical_key, {})
            w_a, w_b, c_ab = la.get("mean"), lb.get("mean"), lc.get("mean")
            if None in (w_a, w_b, c_ab):
                continue
            early_w.append(min(w_a, w_b))
            early_c.append(c_ab)
            lws = [s for s in (la.get("pair_mean_std"), lb.get("pair_mean_std"))
                   if s is not None and s > 0]
            if lws:
                early_s.append(min(lws))
        early_delta = (np.mean(early_w) - np.mean(early_c)) if early_w else None
        early_sigma_delta = (
            (early_delta / np.mean(early_s)) if early_delta is not None and early_s else None
        )

        # Print per-sparsity table
        wa_std_s = f"{within_a_std:.4f}" if within_a_std is not None else "n/a"
        wb_std_s = f"{within_b_std:.4f}" if within_b_std is not None else "n/a"
        sd_s = f"{sigma_delta:+.2f}sigma" if sigma_delta is not None else "n/a"
        esd_s = f"{early_sigma_delta:+.2f}sigma" if early_sigma_delta is not None else "n/a"
        wsd_s = f"{worst_sigma_delta:+.2f}sigma" if worst_sigma_delta is not None else "n/a"
        print(f"\n  sparsity {sparsity}:")
        print(f"    within-{src_a:<12} mean={within_a:.4f}  pair-std={wa_std_s}")
        print(f"    within-{src_b:<12} mean={within_b:.4f}  pair-std={wb_std_s}")
        print(f"    cross  {critical_key:<12} mean={cross_ab:.4f}")
        print(f"    delta:               {delta:+.4f}  ({sd_s})")
        if early_delta is not None:
            print(f"    early-layer (00-07): {early_delta:+.4f}  ({esd_s})")
        print(f"    worst layer:         {worst_layer}  delta={worst_delta:+.4f}  ({wsd_s})")

        summary[spk] = {
            "within_a": within_a, "within_b": within_b,
            "within_a_std": within_a_std, "within_b_std": within_b_std,
            "cross_ab": cross_ab, "baseline": baseline,
            "delta": delta, "sigma_delta": sigma_delta,
            "early_delta": early_delta, "early_sigma_delta": early_sigma_delta,
            "worst_layer": worst_layer, "worst_delta": worst_delta,
            "worst_sigma_delta": worst_sigma_delta,
        }

    print("\n  ----- Interpretation -----")
    if not summary:
        print("  No usable data. Need >=2 seeds per source for the within baseline.")
        return summary

    deltas = [s["delta"] for s in summary.values()]
    sigmas = [s["sigma_delta"] for s in summary.values() if s["sigma_delta"] is not None]
    early_sigmas = [s["early_sigma_delta"] for s in summary.values() if s["early_sigma_delta"] is not None]
    max_delta = max(deltas)
    max_sigma = max(sigmas) if sigmas else None
    max_early_sigma = max(early_sigmas) if early_sigmas else None
    min_baseline = min(s["baseline"] for s in summary.values())

    # Sigma-based thresholds (preferred). Asymmetric: "dead" requires
    # strong evidence (small sigma_delta AND tight within), all other
    # outcomes default to "proceed."
    if max_sigma is None:
        # Sigma unavailable: fall back to absolute thresholds with warning.
        if max_delta < 0.01 and min_baseline > 0.99:
            label = "THESIS MECHANICALLY DEAD (absolute only)"
            body = [
                "Cross-source disagreement is at or below within-source noise,",
                "and within-source is near-perfect. Sigma unavailable - run with",
                ">=2 seeds per source for noise-relative confirmation. Reframe",
                "before Phase 1.",
            ]
        elif max_delta < 0.02:
            label = "EFFECT NOT VISIBLE (absolute only)"
            body = [
                "Run with >=2 seeds per source for noise-relative interpretation.",
                "Proceed with Phase 3 but expect small or null downstream effects.",
            ]
        else:
            label = "EFFECT PRESENT (absolute only)"
            body = [
                f"Largest delta: {max_delta:+.4f}. Sigma unavailable - run with",
                ">=2 seeds for noise-relative confirmation. Proceed with Phase 1+.",
            ]
    elif max_sigma < 0.5 and min_baseline > 0.99:
        label = "THESIS MECHANICALLY DEAD"
        body = [
            "Cross-source disagreement is <0.5 std of within-source noise, and",
            "within-source itself is near-perfect. Wanda would prune nearly",
            "identical weights under either calibration. Downstream eval cannot",
            "find a 'calibration source matters' effect. Reframe before Phase 1.",
        ]
    elif max_sigma < 1.0:
        label = "EFFECT NOT VISIBLE AT MASK LEVEL"
        body = [
            f"Cross-source disagreement peaks at {max_sigma:+.2f} sigma - within",
            "1 std of within-source noise. Effect may still live downstream of",
            "pruning (small mask diffs amplified by later layers / eval), but",
            "Phase 0 alone cannot confirm. Proceed with Phase 3, expect small",
            "or null downstream effects.",
        ]
    elif max_early_sigma is not None and max_early_sigma < 1.0 and max_sigma > 3.0:
        label = "EFFECT IS LATE-LAYER ONLY (caution)"
        body = [
            f"Full sigma: {max_sigma:+.2f}, early-layer sigma: {max_early_sigma:+.2f}.",
            "Early layers (Phase 0's reliable regime) show no cross-vs-within",
            "difference; later layers do. Could be a real cascading effect OR a",
            "Phase 0 artifact from dense-propagation scalers diverging from actual",
            "sequential pruning. Cannot distinguish from masks alone. Phase 3",
            "needed to confirm.",
        ]
    elif max_sigma > 3.0:
        label = "EFFECT PRESENT"
        early_s = f"{max_early_sigma:+.2f}" if max_early_sigma is not None else "n/a"
        body = [
            f"Largest cross-vs-within delta: {max_delta:+.4f} ({max_sigma:+.2f} sigma).",
            f"Early-layer sigma: {early_s}.",
            "Cross-source masks differ from within-source noise by >3 std at",
            "the headline sparsity. Proceed with Phase 1+.",
        ]
    else:
        label = "MARGINAL - PROCEED"
        body = [
            f"Largest sigma: {max_sigma:+.2f} (in the 1-3 std range).",
            "Signal exists but not overwhelming at mask level. Phase 3 will",
            "resolve whether it amplifies downstream.",
        ]

    print(f"  {label}.")
    for line in body:
        print(f"  {line}")

    print()
    print("  Reminder: Phase 0 is a one-way kill switch. 'Effect present' does NOT")
    print("  validate the downstream thesis - only Phase 3 eval runs can. The only")
    print("  verdict Phase 0 renders with confidence is 'thesis mechanically dead'.")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sources", nargs="+", default=["c4", "mc4_ko"],
                        help="Calibration sources to compare.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Seeds per source (>=2 needed for within baseline; 3+ recommended).")
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--critical_pair", type=str, default="c4-mc4_ko",
                        help="Pair 'a-b' for the headline verdict (alphabetical order).")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    parser.add_argument("--save", type=str, required=True)
    args = parser.parse_args()

    if len(args.seeds) < 2:
        print("WARNING: <2 seeds means no within-source baseline.")
        print("         Verdict will be unable to distinguish signal from sample noise.")

    print(f"torch={version('torch')}  transformers={version('transformers')}")
    print(f"loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model or "70b" in args.model:
        device = model.hf_device_map["lm_head"]

    os.makedirs(args.save, exist_ok=True)

    # --- Stage 1: capture scalers for every (source, seed) ---
    captures = {}
    for src in args.sources:
        for sd in args.seeds:
            print(f"\n=== capturing scalers: {src} seed={sd} ===")
            scalers = capture_scalers(model, tokenizer, src, args.nsamples, sd, device)
            captures[(src, sd)] = scalers
            with open(os.path.join(args.save, f"phase0_scalers_{src}_seed{sd}.pkl"), "wb") as f:
                pickle.dump({k: v.numpy() for k, v in scalers.items()}, f)

    # --- Stage 2: per-layer mask analysis ---
    print("\n=== analyzing per-layer masks ===")
    per_layer_stats = {}
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        subset = find_layers(layer)
        for proj_name, linear in subset.items():
            layer_name = f"layer{layer_idx:02d}.{proj_name}"
            per_layer_stats[layer_name] = analyze_layer(linear, layer_name, captures, args.sparsities)
        print(f"  layer {layer_idx:02d} analyzed")

    with open(os.path.join(args.save, "phase0_stats.json"), "w") as f:
        json.dump(per_layer_stats, f, indent=2)

    # --- Stage 3: aggregate ---
    agg_param = aggregate_across_layers(per_layer_stats, args.sparsities, weighting="param")
    agg_unif = aggregate_across_layers(per_layer_stats, args.sparsities, weighting="uniform")

    # --- Stage 4: verdict (uses param-weighted aggregation) ---
    verdict_summary = verdict(per_layer_stats, agg_param, args.critical_pair,
                              args.sources, args.sparsities)

    manifest = {
        "model": args.model,
        "sources": args.sources,
        "seeds": args.seeds,
        "sparsities": args.sparsities,
        "nsamples": args.nsamples,
        "critical_pair": args.critical_pair,
        "torch": version("torch"),
        "transformers": version("transformers"),
        "aggregated_param_weighted": agg_param,
        "aggregated_uniform": agg_unif,
        "verdict_summary": verdict_summary,
    }
    with open(os.path.join(args.save, "phase0_summary.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nsaved scalers, stats, and summary to {args.save}/")


if __name__ == "__main__":
    main()
