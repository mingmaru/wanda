"""
Compare Phase 0's dense-propagation predicted masks against Phase 6's
actual sequential-pruning masks (methodological cross-check).

Phase 0 (phase0.py) captures scaler_row under fully dense activation
propagation and predicts what Wanda WOULD prune. Phase 6
(analyze_masks.py) operates on the masks Wanda actually produced during
sequential pruning of the same (source, seed). The approximation error
grows with layer depth -- Phase 0's docstring explicitly warns about this.

This script quantifies the error:
  For each layer, compare Phase 0's predicted cross-source per-row
  Jaccard vs Phase 6's actual cross-source per-row Jaccard. Both numbers
  describe the same construct: "how much do mask choices differ across
  calibrations at this layer". If they correlate strongly across layers,
  Phase 0's dense-propagation shortcut is a faithful proxy for Phase 3+.
  Where they diverge -- particularly at later layers -- Phase 0 cannot
  substitute for actually running Wanda.

Usage:
    python analyze_phase0_vs_phase6.py \
        --phase0_stats out/phase0_llama3/phase0_stats.json \
        --phase6_csv   out/phase6_c4_vs_mc4ko_50/per_layer_jaccard.csv \
        --pair         c4-mc4_ko \
        --sparsity     0.50 \
        --out          out/alignment_c4_vs_mc4ko_50/

Outputs:
    alignment.csv  per-layer: predicted, actual, residual, proj_type, layer_idx
    alignment.txt  text summary: correlation, MAD, per-projection-type table
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def load_phase0_per_layer(stats_path, pair, sparsity_str):
    """Extract per-layer cross-source predicted Jaccard from phase0_stats.json.

    Returns dict: layer_name -> predicted_jaccard (per-row mean, scalar).
    """
    with open(stats_path) as f:
        stats = json.load(f)
    out = {}
    for layer_name, layer_stats in stats.items():
        block = layer_stats.get(sparsity_str)
        if block is None:
            continue
        cross = (block.get("cross") or {}).get(pair)
        if cross is None:
            continue
        v = cross.get("mean")
        if isinstance(v, (int, float)):
            out[layer_name] = float(v)
    return out


def load_phase6_per_layer(csv_path, pair):
    """Load per-layer actual Jaccards from analyze_masks.py's per_layer_jaccard.csv.

    Filter rows matching --pair. Returns dict: layer_name -> actual_jaccard
    (jaccard_per_row_mean, matching Phase 0's metric).

    The pair name format in analyze_masks.py is 'a__vs__b' (or 'a__VS__b' for
    individual runs); Phase 0 uses 'a-b'. We accept either via fuzzy match
    on the source labels.
    """
    out = {}
    sources = set(pair.split("-"))
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_pair = row.get("pair", "")
            row_sources = set()
            for sep in ("__vs__", "__VS__", "-"):
                if sep in row_pair:
                    row_sources = set(row_pair.split(sep))
                    break
            if row_sources != sources:
                continue
            try:
                jacc = float(row.get("jaccard_per_row_mean", ""))
            except (TypeError, ValueError):
                continue
            out[row["layer"]] = jacc
    return out


def project_type(layer_name):
    parts = layer_name.split(".", 1)
    return parts[1] if len(parts) > 1 else layer_name


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sxx * syy)
    return (sxy / denom) if denom > 0 else None


def mad(xs, ys):
    """Mean absolute deviation between aligned pairs."""
    if not xs:
        return None
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0_stats", required=True,
                        help="Path to phase0_stats.json (from phase0.py).")
    parser.add_argument("--phase6_csv", required=True,
                        help="Path to per_layer_jaccard.csv (from analyze_masks.py).")
    parser.add_argument("--pair", required=True,
                        help="Source pair to compare, e.g. 'c4-mc4_ko' "
                             "(sources alphabetically sorted, as Phase 0 stores them).")
    parser.add_argument("--sparsity", type=float, required=True,
                        help="Sparsity to extract from Phase 0 stats, e.g. 0.5.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_str = f"{args.sparsity:.2f}"
    predicted = load_phase0_per_layer(args.phase0_stats, args.pair, sp_str)
    actual = load_phase6_per_layer(args.phase6_csv, args.pair)

    common = sorted(set(predicted) & set(actual))
    if not common:
        print("No overlapping layer names. Check --pair / --sparsity.")
        print(f"  predicted layers: {len(predicted)} (sample: {list(predicted)[:3]})")
        print(f"  actual layers:    {len(actual)}    (sample: {list(actual)[:3]})")
        return

    rows = []
    for layer in common:
        p = predicted[layer]
        a = actual[layer]
        rows.append({
            "layer": layer,
            "layer_idx": layer.split(".")[0],
            "proj_type": project_type(layer),
            "predicted_jaccard": p,
            "actual_jaccard": a,
            "residual": a - p,  # positive = actual agrees more than predicted
        })

    # Write per-layer CSV
    with open(out_dir / "alignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Overall and per-projection-type stats
    xs = [r["predicted_jaccard"] for r in rows]
    ys = [r["actual_jaccard"] for r in rows]
    r_overall = pearson(xs, ys)
    mad_overall = mad(xs, ys)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["proj_type"]].append(r)

    # Early vs late layer breakdown (layer 0-7 are Phase 0's reliable regime)
    EARLY_PREFIXES = tuple(f"layer{i:02d}" for i in range(8))
    early = [r for r in rows if r["layer"].startswith(EARLY_PREFIXES)]
    late = [r for r in rows if not r["layer"].startswith(EARLY_PREFIXES)]

    def summarize(label, rs):
        if not rs:
            return f"  {label}: no data"
        xs = [r["predicted_jaccard"] for r in rs]
        ys = [r["actual_jaccard"] for r in rs]
        return (f"  {label:<28} n={len(rs):>3}  "
                f"pred_mean={sum(xs)/len(xs):.4f}  "
                f"actual_mean={sum(ys)/len(ys):.4f}  "
                f"MAD={mad(xs, ys):.4f}  "
                f"Pearson r={pearson(xs, ys) if len(xs) >= 2 else 'n/a'}")

    lines = []
    lines.append(f"Phase 0 vs Phase 6 alignment for pair={args.pair} sparsity={args.sparsity}")
    lines.append(f"  phase0_stats: {args.phase0_stats}")
    lines.append(f"  phase6_csv:   {args.phase6_csv}")
    lines.append(f"  layers compared: {len(rows)}")
    lines.append("")
    lines.append(f"Overall Pearson r = {r_overall:.4f}" if r_overall is not None else "Overall r = n/a")
    lines.append(f"Overall MAD       = {mad_overall:.4f}" if mad_overall is not None else "Overall MAD = n/a")
    lines.append("")
    lines.append("By layer depth:")
    lines.append(summarize("early (layers 00-07)", early))
    lines.append(summarize("late  (layers 08+)", late))
    lines.append("")
    lines.append("By projection type:")
    for proj, rs in sorted(grouped.items()):
        lines.append(summarize(proj, rs))
    lines.append("")
    lines.append("Interpretation:")
    if r_overall is None:
        lines.append("  Too few layers compared to compute correlation.")
    elif r_overall > 0.9:
        lines.append("  STRONG ALIGNMENT (r > 0.9). Phase 0 is a faithful proxy for the")
        lines.append("  cross-source mask differences Wanda actually produces. The dense-")
        lines.append("  propagation approximation does not materially distort the verdict.")
    elif r_overall > 0.6:
        lines.append("  MODERATE ALIGNMENT (0.6 < r < 0.9). Phase 0 captures most of the")
        lines.append("  signal but loses some accuracy, especially at later layers. Cite")
        lines.append("  the early-layer subset when discussing Phase 0 verdicts.")
    else:
        lines.append("  WEAK ALIGNMENT (r < 0.6). Phase 0's dense-propagation predictions")
        lines.append("  diverge substantially from actual sequential pruning. Phase 0 should")
        lines.append("  be cited as a necessary-condition test only; do not generalize its")
        lines.append("  verdict to claims about Phase 3 effects.")

    with open(out_dir / "alignment.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
