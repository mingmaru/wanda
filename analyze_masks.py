"""
Phase 6 mask analysis: compare pruning masks across runs.

For each pair of runs (or pair of run-groups), compute per-layer and
per-projection-type Jaccard between their saved boolean pruning masks.
Aggregate across layers (param-weighted and unweighted). Output CSV +
LaTeX table for the report.

Two operating modes:

  --runs path1 path2 [path3 ...]
      Pairwise compare every run with every other run.

  --runs_by_source label1=path1,path2,...  label2=path3,...
      Group runs by source label. For each cross-label pair, average the
      per-layer Jaccards across all (run_a, run_b) cross-pairs. Use this
      when each label has multiple seeds and you want the across-seed
      mean (matches the predictions.md analysis decisions).

Glob patterns are supported in the per-path lists. Each path is a run
save directory containing masks_{method}.pt produced by lib/prune.py.

Outputs three files in --out:
    per_layer_jaccard.csv     -- one row per (pair, layer)
    per_proj_type.csv         -- aggregated within projection type
    summary.tex               -- LaTeX table for the report
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch


def load_masks(run_dir, method):
    """Load the saved mask dict from one run directory.

    Returns dict of layer_name -> bool tensor (CPU).
    """
    path = Path(run_dir) / f"masks_{method}.pt"
    if not path.exists():
        raise FileNotFoundError(f"no mask file at {path}")
    masks = torch.load(path, map_location="cpu")
    # The bool tensors are saved as torch.bool; no conversion needed.
    return masks


def jaccard_layer(a, b):
    """Layer-wide Jaccard on two bool tensors of equal shape.

    inter / union over all entries. With identical sparsity ratios per
    layer, |union| = 2K - |inter| where K = number of pruned entries.
    """
    inter = int((a & b).sum().item())
    union = int((a | b).sum().item())
    return (inter / union) if union > 0 else 1.0


def jaccard_per_row_mean(a, b):
    """Mean of per-row Jaccards (consistent with phase0.py's metric)."""
    inter = (a & b).sum(dim=1).float()
    union = (a | b).sum(dim=1).float()
    per_row = torch.where(union > 0, inter / union, torch.ones_like(inter))
    return float(per_row.mean().item())


def project_type(layer_name):
    """'layer05.self_attn.q_proj' -> 'self_attn.q_proj'."""
    parts = layer_name.split(".", 1)
    return parts[1] if len(parts) > 1 else layer_name


def compute_pair_jaccards(masks_a, masks_b):
    """Compute per-layer Jaccards for one mask-dict pair."""
    rows = []
    keys = sorted(set(masks_a.keys()) & set(masks_b.keys()))
    for k in keys:
        ma, mb = masks_a[k], masks_b[k]
        if ma.shape != mb.shape:
            print(f"  WARN: shape mismatch on {k}: {ma.shape} vs {mb.shape}; skipping")
            continue
        rows.append({
            "layer": k,
            "proj_type": project_type(k),
            "layer_idx": k.split(".")[0],
            "n_params": int(ma.numel()),
            "jaccard_layer": jaccard_layer(ma, mb),
            "jaccard_per_row_mean": jaccard_per_row_mean(ma, mb),
        })
    return rows


def aggregate_by_proj_type(per_layer_rows):
    """Within each projection type, compute param-weighted and unweighted
    means of the layer-Jaccard, plus mean of per-row-mean across layers.
    """
    grouped = defaultdict(list)
    for row in per_layer_rows:
        grouped[row["proj_type"]].append(row)
    summary = []
    for proj, rows in sorted(grouped.items()):
        total = sum(r["n_params"] for r in rows)
        pw = sum(r["n_params"] * r["jaccard_layer"] for r in rows) / total if total else 0.0
        uw = sum(r["jaccard_layer"] for r in rows) / len(rows)
        prm = sum(r["jaccard_per_row_mean"] for r in rows) / len(rows)
        summary.append({
            "proj_type": proj,
            "n_layers": len(rows),
            "total_params": total,
            "jaccard_layer_paramweighted": pw,
            "jaccard_layer_unweighted": uw,
            "jaccard_per_row_mean": prm,
        })
    return summary


def write_per_layer_csv(rows_by_pair, out_path):
    fieldnames = ["pair", "layer", "proj_type", "layer_idx", "n_params",
                  "jaccard_layer", "jaccard_per_row_mean"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pair, rows in rows_by_pair.items():
            for r in rows:
                w.writerow({**r, "pair": pair})


def write_proj_type_csv(summary_by_pair, out_path):
    fieldnames = ["pair", "proj_type", "n_layers", "total_params",
                  "jaccard_layer_paramweighted", "jaccard_layer_unweighted",
                  "jaccard_per_row_mean"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pair, rows in summary_by_pair.items():
            for r in rows:
                w.writerow({**r, "pair": pair})


def write_latex_table(summary_by_pair, out_path):
    """LaTeX table: rows = projection types, columns = pairs.

    Cells are the param-weighted Jaccard (the primary metric we'd
    cite in the report; per_row_mean is in the CSVs for detail).
    """
    pairs = list(summary_by_pair.keys())
    proj_types_set = set()
    for rows in summary_by_pair.values():
        for r in rows:
            proj_types_set.add(r["proj_type"])
    proj_types = sorted(proj_types_set)

    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(pairs) + "}")
    lines.append("\\toprule")
    header = ["Projection"] + [p.replace("_", "\\_") for p in pairs]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for pt in proj_types:
        cells = [pt.replace("_", "\\_")]
        for pair in pairs:
            lookup = {r["proj_type"]: r["jaccard_layer_paramweighted"]
                      for r in summary_by_pair[pair]}
            v = lookup.get(pt)
            cells.append(f"{v:.3f}" if v is not None else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_runs_by_source(specs):
    """Parse --runs_by_source 'label=path1,path2,...' entries.

    Each entry maps a source label (e.g. 'c4', 'mc4_ko') to one or more
    run directories. Glob patterns are expanded.
    """
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"expected label=paths, got '{spec}'")
        label, _, path_str = spec.partition("=")
        expanded = []
        for p in path_str.split(","):
            matches = sorted(glob.glob(p))
            expanded.extend(matches if matches else [p])
        result[label.strip()] = expanded
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+",
                        help="Run directories to compare pairwise (every pair).")
    parser.add_argument("--runs_by_source", nargs="+",
                        help="'label=path1,path2,...' groups runs by source. Cross-label "
                             "pairs are averaged across all member-pair combinations.")
    parser.add_argument("--method", required=True,
                        choices=["wanda", "magnitude", "sparsegpt"],
                        help="Which masks_<method>.pt file to read from each run dir.")
    parser.add_argument("--out", required=True,
                        help="Output directory for CSVs and LaTeX.")
    args = parser.parse_args()

    if not args.runs and not args.runs_by_source:
        parser.error("need --runs or --runs_by_source")

    os.makedirs(args.out, exist_ok=True)

    rows_by_pair = {}
    summary_by_pair = {}

    if args.runs:
        runs = args.runs
        labels = [Path(r).name for r in runs]
        for (i, ri), (j, rj) in combinations(enumerate(runs), 2):
            pair_name = f"{labels[i]}__vs__{labels[j]}"
            print(f"\n=== {pair_name} ===")
            ma = load_masks(ri, args.method)
            mb = load_masks(rj, args.method)
            rows = compute_pair_jaccards(ma, mb)
            rows_by_pair[pair_name] = rows
            summary_by_pair[pair_name] = aggregate_by_proj_type(rows)
            print(f"  {len(rows)} layers compared")
            for s in summary_by_pair[pair_name]:
                print(f"    {s['proj_type']:>25}: layer-Jacc(pw)={s['jaccard_layer_paramweighted']:.4f}  "
                      f"per-row-mean={s['jaccard_per_row_mean']:.4f}")

    if args.runs_by_source:
        groups = parse_runs_by_source(args.runs_by_source)
        # Preload all masks to avoid reloading inside nested loops.
        mask_cache = {}
        for label, paths in groups.items():
            for p in paths:
                if p not in mask_cache:
                    mask_cache[p] = load_masks(p, args.method)

        sources = sorted(groups.keys())
        for sa, sb in combinations(sources, 2):
            pair_name = f"{sa}__vs__{sb}"
            print(f"\n=== {pair_name} ===")
            # For each (run_a in sa, run_b in sb), compute per-layer Jaccards.
            # Then average per-layer across all member-pair combinations.
            by_layer = defaultdict(list)
            for ra in groups[sa]:
                for rb in groups[sb]:
                    for r in compute_pair_jaccards(mask_cache[ra], mask_cache[rb]):
                        by_layer[r["layer"]].append(r)
            avg_rows = []
            for layer_name, rs in sorted(by_layer.items()):
                avg_rows.append({
                    "layer": layer_name,
                    "proj_type": rs[0]["proj_type"],
                    "layer_idx": rs[0]["layer_idx"],
                    "n_params": rs[0]["n_params"],
                    "jaccard_layer": sum(r["jaccard_layer"] for r in rs) / len(rs),
                    "jaccard_per_row_mean": sum(r["jaccard_per_row_mean"] for r in rs) / len(rs),
                })
            rows_by_pair[pair_name] = avg_rows
            summary_by_pair[pair_name] = aggregate_by_proj_type(avg_rows)
            print(f"  {len(avg_rows)} layers, averaged over {len(groups[sa])} x {len(groups[sb])} run pairs")
            for s in summary_by_pair[pair_name]:
                print(f"    {s['proj_type']:>25}: layer-Jacc(pw)={s['jaccard_layer_paramweighted']:.4f}  "
                      f"per-row-mean={s['jaccard_per_row_mean']:.4f}")

    write_per_layer_csv(rows_by_pair, os.path.join(args.out, "per_layer_jaccard.csv"))
    write_proj_type_csv(summary_by_pair, os.path.join(args.out, "per_proj_type.csv"))
    write_latex_table(summary_by_pair, os.path.join(args.out, "summary.tex"))
    print(f"\nwrote results to {args.out}/")


if __name__ == "__main__":
    main()
