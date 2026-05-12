"""
Aggregate Phase 3 run results and compute the pre-registered F-ratio.

Walks --runs_dir for manifest.json files (produced by main.py write_manifest).
Flattens each manifest into one CSV row with all relevant metrics. Then
groups by (sparsity_ratio, metric) and computes the variance-ratio F-test
that predictions.md section 4 pre-registered as the primary statistical
test:

    F = MS_between / MS_within
        where MS_between = Var across calibration-source means (weighted by n_g)
              MS_within  = Pooled within-source variance across seeds

This is the standard one-way ANOVA F-statistic, accommodating unequal
seed counts per source (predictions.md section 7 specifies 5 seeds on
headline cells and 3 on secondary cells).

Predictions.md verdict thresholds (section 9):
    F < 1.5   -> thesis falsified
    F > 3.0   -> thesis supported (at this metric/sparsity)
    1.5 - 3.0 -> marginal, defer to other evidence

Usage:
    python aggregate.py --runs_dir out/phase3/ --out out/aggregated/

Output:
    summary.csv     one row per cell (calib_data, sparsity, seed, all metrics)
    f_ratios.csv    one row per (sparsity, metric) with F, df, verdict
    f_ratios.tex    LaTeX table for the report
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# Each metric-extractor walks the manifest's results structure and returns a
# scalar (or None if unavailable). Robust to the slight key variation across
# lm-eval-harness 0.4.x point releases ('acc,none' vs 'acc' etc.).

def _find_downstream(manifest, outer_key, inner_key, candidate_keys):
    downstream = (manifest.get("results") or {}).get("downstream") or {}
    outer = downstream.get(outer_key) or {}
    inner = outer.get(inner_key) or {}
    for k in candidate_keys:
        v = inner.get(k)
        if isinstance(v, (int, float)):
            return v
    return None


METRICS_TO_EXTRACT = {
    "ppl_wikitext2": lambda m: (m.get("results") or {}).get("ppl_wikitext2"),
    "ppl_mc4_ko":    lambda m: (m.get("results") or {}).get("ppl_mc4_ko"),
    "nlu_bundle_avg": lambda m: (m.get("results") or {}).get("nlu_bundle_avg"),
    "mmlu_acc":      lambda m: _find_downstream(m, "mmlu", "mmlu", ["acc,none", "acc"]),
    "gsm8k_acc":     lambda m: _find_downstream(m, "gsm8k", "gsm8k",
                                                ["exact_match,strict-match",
                                                 "exact_match,flexible-extract",
                                                 "exact_match", "acc,none", "acc"]),
    "kobest_hellaswag_acc": lambda m: _find_downstream(m, "kobest_hellaswag",
                                                       "kobest_hellaswag",
                                                       ["acc,none", "acc"]),
    "kmmlu_acc":     lambda m: _find_downstream(m, "kmmlu", "kmmlu", ["acc,none", "acc"]),
}

ROW_FIELDS = [
    "cell_dir", "model", "prune_method", "calib_data",
    "sparsity_ratio_requested", "sparsity_ratio_actual",
    "seed", "nsamples", "override_shot",
] + list(METRICS_TO_EXTRACT.keys())


def load_manifests(runs_dir):
    """Yield (manifest_path, manifest_dict) for every manifest under runs_dir."""
    for path in sorted(Path(runs_dir).glob("**/manifest.json")):
        try:
            with open(path) as f:
                yield path, json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: could not load {path}: {e}")


def flatten_manifest(manifest, source_path):
    """Manifest dict -> flat row dict matching ROW_FIELDS."""
    args = manifest.get("args", {}) or {}
    row = {
        "cell_dir": str(source_path.parent),
        "model": args.get("model"),
        "prune_method": args.get("prune_method"),
        "calib_data": args.get("calib_data"),
        "sparsity_ratio_requested": args.get("sparsity_ratio_requested"),
        "sparsity_ratio_actual": manifest.get("sparsity_ratio_actual"),
        "seed": args.get("seed"),
        "nsamples": args.get("nsamples"),
        "override_shot": args.get("override_shot"),
    }
    for name, getter in METRICS_TO_EXTRACT.items():
        try:
            row[name] = getter(manifest)
        except (KeyError, TypeError):
            row[name] = None
    return row


def one_way_anova(groups):
    """One-way ANOVA F for a dict of {group_name: list of scalar observations}.

    Handles unbalanced designs (different n_g across groups). Filters None/NaN
    observations. Returns None if fewer than 2 groups have any data or if
    N <= k (no within-group degrees of freedom).
    """
    cleaned = {}
    for g, vals in groups.items():
        filtered = [float(v) for v in vals
                    if v is not None and not (isinstance(v, float) and v != v)]
        if filtered:
            cleaned[g] = filtered

    if len(cleaned) < 2:
        return None
    all_vals = [v for vs in cleaned.values() for v in vs]
    N = len(all_vals)
    k = len(cleaned)
    if N <= k:
        return None

    grand_mean = sum(all_vals) / N
    group_means = {g: sum(vs) / len(vs) for g, vs in cleaned.items()}

    ss_between = sum(len(vs) * (group_means[g] - grand_mean) ** 2
                     for g, vs in cleaned.items())
    ss_within = sum(sum((v - group_means[g]) ** 2 for v in vs)
                    for g, vs in cleaned.items())

    df_between = k - 1
    df_within = N - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    if ms_within > 0:
        F = ms_between / ms_within
    elif ms_between > 0:
        F = float("inf")
    else:
        F = float("nan")

    return {
        "F": F,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "group_means": group_means,
        "group_ns": {g: len(vs) for g, vs in cleaned.items()},
        "grand_mean": grand_mean,
        "n_total": N,
        "k_groups": k,
    }


def verdict_for_F(F):
    """predictions.md section 9: thresholds at 1.5 and 3.0."""
    if F is None or F != F:  # None or NaN
        return "INSUFFICIENT_DATA"
    if F < 1.5:
        return "FALSIFIED_F<1.5"
    if F > 3.0:
        return "SUPPORTS_F>3"
    return "MARGINAL_1.5-3"


def write_summary_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ROW_FIELDS})


def write_f_ratios_csv(f_rows, out_path):
    all_cols = set()
    for r in f_rows:
        all_cols.update(r.keys())
    std_first = ["sparsity_ratio", "metric", "F", "df_between", "df_within",
                 "verdict", "n_total", "k_groups", "ms_between", "ms_within"]
    ordered = std_first + sorted(all_cols - set(std_first))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in f_rows:
            w.writerow(r)


def write_f_ratios_latex(f_rows, out_path):
    """Pivot: rows = metric, columns = sparsity. Cells are 'F (verdict_marker)'."""
    if not f_rows:
        with open(out_path, "w") as f:
            f.write("% no data\n")
        return

    metrics = sorted({r["metric"] for r in f_rows})
    sparsities = sorted({r["sparsity_ratio"] for r in f_rows
                         if r["sparsity_ratio"] is not None})
    lookup = {(r["metric"], r["sparsity_ratio"]): r for r in f_rows}

    lines = ["\\begin{tabular}{l" + "c" * len(sparsities) + "}", "\\toprule"]
    header = ["Metric"] + [f"{int(s * 100)}\\%" for s in sparsities]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for m in metrics:
        cells = [m.replace("_", "\\_")]
        for s in sparsities:
            r = lookup.get((m, s))
            if r is None or r.get("F") is None:
                cells.append("--")
            else:
                F = r["F"]
                marker = ""
                if r["verdict"] == "SUPPORTS_F>3":
                    marker = "$^{*}$"
                elif r["verdict"] == "FALSIFIED_F<1.5":
                    marker = "$^{\\dagger}$"
                if F == float("inf"):
                    cells.append(f"$\\infty${marker}")
                elif F != F:
                    cells.append("nan")
                else:
                    cells.append(f"{F:.2f}{marker}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("% Notes: F > 3 (*) supports thesis; F < 1.5 (dagger) falsifies. "
                 "See predictions.md section 9 for the pre-registered thresholds.")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def aggregate(runs_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, manifest in load_manifests(runs_dir):
        rows.append(flatten_manifest(manifest, path))
    print(f"loaded {len(rows)} manifests from {runs_dir}")

    if not rows:
        return

    write_summary_csv(rows, out_dir / "summary.csv")
    print(f"wrote {out_dir / 'summary.csv'}")

    # Group by (sparsity_ratio, metric, calib_data) and compute F per
    # (sparsity_ratio, metric) -- treating calib_data values as the
    # ANOVA groups, seeds as the within-group observations.
    f_rows = []
    sparsities = sorted({r["sparsity_ratio_requested"] for r in rows
                         if r["sparsity_ratio_requested"] is not None})
    for sp in sparsities:
        if sp == 0:
            continue  # dense has no calibration → no F to compute
        rows_sp = [r for r in rows if r["sparsity_ratio_requested"] == sp]
        for metric in METRICS_TO_EXTRACT:
            groups = defaultdict(list)
            for r in rows_sp:
                src = r.get("calib_data")
                v = r.get(metric)
                if src is not None and v is not None:
                    groups[src].append(v)
            result = one_way_anova(dict(groups))
            row = {
                "sparsity_ratio": sp,
                "metric": metric,
                "F": result["F"] if result else None,
                "df_between": result["df_between"] if result else None,
                "df_within": result["df_within"] if result else None,
                "verdict": verdict_for_F(result["F"]) if result else "INSUFFICIENT_DATA",
                "n_total": result["n_total"] if result else 0,
                "k_groups": result["k_groups"] if result else 0,
                "ms_between": result["ms_between"] if result else None,
                "ms_within": result["ms_within"] if result else None,
            }
            if result:
                for g, mean in result["group_means"].items():
                    row[f"mean_{g}"] = mean
                for g, n in result["group_ns"].items():
                    row[f"n_{g}"] = n
            f_rows.append(row)

    write_f_ratios_csv(f_rows, out_dir / "f_ratios.csv")
    print(f"wrote {out_dir / 'f_ratios.csv'}")
    write_f_ratios_latex(f_rows, out_dir / "f_ratios.tex")
    print(f"wrote {out_dir / 'f_ratios.tex'}")

    print("\n=== predictions.md verdict against F-ratio thresholds ===")
    for r in f_rows:
        if r["F"] is None:
            continue
        f_str = f"{r['F']:8.3f}" if r["F"] != float("inf") else "     inf"
        print(f"  sparsity={r['sparsity_ratio']:.2f}  {r['metric']:<22} "
              f"F={f_str}  df=({r['df_between']},{r['df_within']})  "
              f"verdict={r['verdict']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True,
                        help="Directory containing run sub-directories with manifest.json files.")
    parser.add_argument("--out", required=True, help="Output directory for CSVs and LaTeX.")
    args = parser.parse_args()
    aggregate(args.runs_dir, args.out)


if __name__ == "__main__":
    main()
