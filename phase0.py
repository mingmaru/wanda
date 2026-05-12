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
    every pair of (source, seed) captures. Returns nested dict.
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

        within = defaultdict(list)   # source -> list of per-row jaccard arrays
        cross = defaultdict(list)    # "src_a-src_b" sorted -> list

        for e1, e2 in combinations(entities, 2):
            j = per_row_jaccard(masks[e1], masks[e2]).numpy()
            src1, _ = e1
            src2, _ = e2
            if src1 == src2:
                within[src1].append(j)
            else:
                key = "-".join(sorted([src1, src2]))
                cross[key].append(j)

        results[spk] = {
            "within": {src: jaccard_stats(np.concatenate(arr)) for src, arr in within.items()},
            "cross": {key: jaccard_stats(np.concatenate(arr)) for key, arr in cross.items()},
        }

        # Free per-sparsity masks before next sparsity
        del masks

    return results


# ============================================================
# Stage 3: aggregate across layers
# ============================================================

def aggregate_across_layers(per_layer_stats, sparsities, weighting):
    """Aggregate per-layer means into model-level numbers.

    weighting="param" weights each layer by d_out * d_in (matches Wanda's
    actual operation scope). "uniform" weights each layer equally.
    """
    aggregated = {}
    for sparsity in sparsities:
        spk = f"{sparsity:.2f}"
        within_keys = set()
        cross_keys = set()
        for layer_stats in per_layer_stats.values():
            within_keys.update(layer_stats[spk]["within"].keys())
            cross_keys.update(layer_stats[spk]["cross"].keys())

        within_agg = {k: {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0} for k in within_keys}
        cross_agg = {k: {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0} for k in cross_keys}
        total_weight = 0.0

        for layer_stats in per_layer_stats.values():
            w = layer_stats["params"] if weighting == "param" else 1.0
            total_weight += w
            for k, st in layer_stats[spk]["within"].items():
                for m in ["mean", "p10", "p50", "p90"]:
                    within_agg[k][m] += st[m] * w
            for k, st in layer_stats[spk]["cross"].items():
                for m in ["mean", "p10", "p50", "p90"]:
                    cross_agg[k][m] += st[m] * w

        for d in (within_agg, cross_agg):
            for k in d:
                for m in d[k]:
                    d[k][m] /= total_weight

        aggregated[spk] = {"within": within_agg, "cross": cross_agg}
    return aggregated


# ============================================================
# Stage 4: verdict
# ============================================================

EARLY_LAYER_PREFIXES = tuple(f"layer{i:02d}" for i in range(8))


def verdict(per_layer_stats, aggregated, critical_pair, sources, sparsities):
    """Render headline verdict for the critical cross-source pair.

    Reports cross - within delta at each sparsity. The within-source
    baseline is the *tighter* of the two source baselines (the more
    conservative comparison). Also reports early-layer-only delta since
    Phase 0's dense-propagation approximation is more accurate there.
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
        within_a = aggregated[spk]["within"].get(src_a, {}).get("mean")
        within_b = aggregated[spk]["within"].get(src_b, {}).get("mean")
        cross_ab = aggregated[spk]["cross"].get(critical_key, {}).get("mean")

        if None in (within_a, within_b, cross_ab):
            print(f"  sparsity {sparsity}: missing data (need >= 2 seeds per source).")
            continue

        baseline = min(within_a, within_b)
        delta = baseline - cross_ab  # positive = cross worse than within

        # Worst layer for this pair
        worst_layer, worst_delta = None, -1e9
        for ln, ls in per_layer_stats.items():
            w_a = ls[spk]["within"].get(src_a, {}).get("mean")
            w_b = ls[spk]["within"].get(src_b, {}).get("mean")
            c_ab = ls[spk]["cross"].get(critical_key, {}).get("mean")
            if None in (w_a, w_b, c_ab):
                continue
            d = min(w_a, w_b) - c_ab
            if d > worst_delta:
                worst_delta, worst_layer = d, ln

        # Early-layer aggregate (Phase 0's reliable regime)
        early_within, early_cross = [], []
        for ln, ls in per_layer_stats.items():
            if not ln.startswith(EARLY_LAYER_PREFIXES):
                continue
            w_a = ls[spk]["within"].get(src_a, {}).get("mean")
            w_b = ls[spk]["within"].get(src_b, {}).get("mean")
            c_ab = ls[spk]["cross"].get(critical_key, {}).get("mean")
            if None in (w_a, w_b, c_ab):
                continue
            early_within.append(min(w_a, w_b))
            early_cross.append(c_ab)
        early_delta = (np.mean(early_within) - np.mean(early_cross)) if early_within else None

        print(f"\n  sparsity {sparsity}:")
        print(f"    within-{src_a:<10} mean Jaccard:  {within_a:.4f}")
        print(f"    within-{src_b:<10} mean Jaccard:  {within_b:.4f}")
        print(f"    cross  {critical_key:<10}      :  {cross_ab:.4f}")
        print(f"    baseline (tighter within)    :  {baseline:.4f}")
        print(f"    delta (baseline - cross)     : {delta:+.4f}")
        if early_delta is not None:
            print(f"    early-layer delta (00-07)    : {early_delta:+.4f}")
        print(f"    worst layer                  :  {worst_layer} (delta {worst_delta:+.4f})")

        summary[spk] = {
            "within_a": within_a, "within_b": within_b,
            "cross_ab": cross_ab, "baseline": baseline, "delta": delta,
            "early_delta": early_delta,
            "worst_layer": worst_layer, "worst_delta": worst_delta,
        }

    print("\n  ----- Interpretation -----")
    if not summary:
        print("  No usable data. Need >= 2 seeds per source for the within baseline.")
        return summary

    deltas = [s["delta"] for s in summary.values()]
    max_delta = max(deltas)
    min_baseline = min(s["baseline"] for s in summary.values())
    early_deltas = [s["early_delta"] for s in summary.values() if s["early_delta"] is not None]
    max_early = max(early_deltas) if early_deltas else None

    # Thresholds are heuristic and asymmetric: "dead" requires strong
    # evidence (high cross agreement AND tight within), "proceed" is
    # the default.
    if max_delta < 0.01 and min_baseline > 0.99:
        print("  THESIS MECHANICALLY DEAD.")
        print("  Cross-source mask disagreement is at or below within-source noise,")
        print("  and within-source itself is near-perfect. Wanda would prune nearly")
        print("  identical weights under either calibration. Downstream eval cannot")
        print("  find a 'calibration source matters' effect. Reframe before Phase 1.")
    elif max_delta < 0.02:
        print("  EFFECT NOT VISIBLE AT MASK LEVEL.")
        print("  Within-source noise is comparable to cross-source. The effect may")
        print("  still live downstream of pruning (small mask diffs amplified by")
        print("  later layers / eval), but Phase 0 alone cannot confirm. Proceed")
        print("  with Phase 3 but expect small or null downstream effects.")
    elif max_early is not None and max_early < 0.02 and max_delta > 0.05:
        print("  EFFECT IS LATE-LAYER ONLY (interpret with caution).")
        print("  Early layers (Phase 0's reliable regime) show no cross-vs-within")
        print("  difference; later layers do. This could be a real cascading effect")
        print("  OR a Phase 0 artifact from dense-propagation scalers diverging")
        print("  from actual sequential pruning. Phase 0 cannot distinguish.")
        print("  Proceed with Phase 3 to confirm.")
    else:
        sp_max = [k for k, v in summary.items() if v["delta"] == max_delta][0]
        print("  EFFECT PRESENT. Proceed with Phase 1+.")
        print(f"  Largest cross-vs-within delta: {max_delta:+.4f} at sparsity {sp_max}.")
        if max_early is not None:
            print(f"  Early-layer delta (most reliable): {max_early:+.4f}.")

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
