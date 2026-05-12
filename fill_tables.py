"""
Generate LaTeX table fragments for the report from the aggregated summary.

Reads the summary.csv produced by aggregate.py and emits one tabular
fragment per logical table in final_report/main.tex. The user either
copy-pastes the fragment into main.tex or replaces the inline table
with `\\input{tables/<name>.tex}`.

Provides:
  - exp1_calib_domain: calibration source as rows, metrics as columns,
                       filtered to a chosen sparsity (default 0.5).
  - exp3_calib_x_sparsity: (calibration, sparsity) grid, headline metrics.
  - generic pivot: --row_var X --col_var Y --filter "key=val,key=val"

Each non-dense cell becomes "mean +/- std" across seeds for that
(calib_data, sparsity, ...) combination. Sparsity=0 rows are emitted
as "Dense" rows (single observation, no std).

Usage:
    python fill_tables.py --summary out/aggregated/summary.csv \\
        --out final_report/tables/
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

# Default column lineup for the headline calibration-domain tables.
# Order matters: WikiText-2 ppl first (anchor), then English knowledge
# (MMLU), English math (GSM8K), then the two Korean tasks plus the
# Korean ppl (the continuous metric that has the cleanest signal).
HEADLINE_METRICS = [
    ("ppl_wikitext2", "WT-2 ppl", "down"),
    ("mmlu_acc", "MMLU", "up"),
    ("gsm8k_acc", "GSM8K", "up"),
    ("kobest_hellaswag_acc", "KoBEST HS", "up"),
    ("kmmlu_acc", "KMMLU", "up"),
    ("ppl_mc4_ko", "MC4-ko ppl", "down"),
]


# ---------- I/O ----------

def load_summary(summary_path):
    """summary.csv -> list of dicts with values coerced to float where possible."""
    rows = []
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in list(r.items()):
                if v in ("", "None", None):
                    r[k] = None
                else:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass  # leave as string
            rows.append(r)
    return rows


def fmt_cell(val, std=None, decimals=4):
    """Render one cell: 'mean +/- std' or '--' if missing.

    Perplexities use 2 decimal places by convention (much wider range).
    """
    if val is None or (isinstance(val, float) and val != val):
        return "--"
    if std is None or std == 0:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def fmt_metric_value(val, std, metric_key):
    """Metric-aware formatter (perplexity vs accuracy)."""
    if val is None:
        return "--"
    if "ppl" in metric_key:
        return fmt_cell(val, std, decimals=2)
    # Accuracies as percentages with 2 decimals
    val_pct = val * 100
    std_pct = std * 100 if std is not None else None
    if std_pct is None or std_pct == 0:
        return f"{val_pct:.2f}"
    return f"{val_pct:.2f} $\\pm$ {std_pct:.2f}"


# ---------- aggregation primitives ----------

def mean_std(vals):
    """Sample mean and (n-1) std of a list. Returns (mean, std). 0 std if n=1."""
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def group_rows(rows, *keys):
    """Group rows by tuple of values at given keys. Returns dict tuple -> list[row]."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r.get(k) for k in keys)].append(r)
    return grouped


# ---------- table generators ----------

def write_exp1_calib_domain(rows, out_path, sparsity=0.5, model_filter=None):
    """Calibration-source effect at a chosen sparsity.

    Rows: Magnitude / SparseGPT(C4) / Wanda(C4) / Wanda(MC4-ko) /
          Wanda(The Stack) / Wanda(random). Optionally a Dense row at the
          top if sparsity=0 rows exist.
    Columns: HEADLINE_METRICS.
    """
    rows = [r for r in rows
            if (model_filter is None or r.get("model") == model_filter)]
    dense_rows = [r for r in rows if r.get("sparsity_ratio_requested") == 0]
    sp_rows = [r for r in rows if r.get("sparsity_ratio_requested") == sparsity]

    row_specs = []
    if dense_rows:
        row_specs.append(("Dense (no pruning)", None, None, dense_rows))
    # Method+calibration combinations expected at the headline sparsity.
    row_specs.extend([
        ("Magnitude", "magnitude", None, sp_rows),
        ("SparseGPT (C4)", "sparsegpt", "c4", sp_rows),
        ("Wanda (C4)", "wanda", "c4", sp_rows),
        ("Wanda (MC4-ko)", "wanda", "mc4_ko", sp_rows),
        ("Wanda (The Stack)", "wanda", "the_stack", sp_rows),
        ("Wanda (random)", "wanda", "random", sp_rows),
    ])

    lines = []
    n_cols = 1 + len(HEADLINE_METRICS)
    col_spec = "l" + "c" * len(HEADLINE_METRICS)
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    headers = ["Method (cal.)"]
    for _, label, direction in HEADLINE_METRICS:
        arrow = "$\\downarrow$" if direction == "down" else "$\\uparrow$"
        headers.append(f"{label} {arrow}")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for label, method, calib, src_rows in row_specs:
        match = [r for r in src_rows
                 if (method is None or r.get("prune_method") == method)
                 and (calib is None or r.get("calib_data") == calib)]
        cells = [label]
        for metric_key, _, _ in HEADLINE_METRICS:
            vals = [r.get(metric_key) for r in match]
            m, s = mean_std(vals)
            cells.append(fmt_metric_value(m, s, metric_key))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"% Generated by fill_tables.py from summary.csv "
                 f"at sparsity={sparsity}; n_cells={sum(len(s[3]) for s in row_specs)}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_exp3_calib_x_sparsity(rows, out_path, model_filter=None):
    """Calibration x sparsity interaction for the two headline calibrations.

    Rows: C4 50%, C4 70%, MC4-ko 50%, MC4-ko 70%. Optionally Dense row.
    Columns: MMLU, KoBEST-HS, KMMLU, MC4-ko ppl with delta-vs-dense if
             dense baseline is present.
    """
    rows = [r for r in rows
            if (model_filter is None or r.get("model") == model_filter)]
    dense = [r for r in rows if r.get("sparsity_ratio_requested") == 0]

    interaction_metrics = [
        ("mmlu_acc", "MMLU", "up"),
        ("kobest_hellaswag_acc", "KoBEST HS", "up"),
        ("kmmlu_acc", "KMMLU", "up"),
        ("ppl_mc4_ko", "MC4-ko ppl", "down"),
    ]

    # Compute dense reference per metric.
    dense_ref = {}
    for k, _, _ in interaction_metrics:
        m, _ = mean_std([r.get(k) for r in dense])
        dense_ref[k] = m

    row_specs = []
    if dense:
        row_specs.append(("Dense", None, None, dense))
    for calib in ("c4", "mc4_ko"):
        for sp in (0.5, 0.7):
            label = f"{calib.replace('_', '-')} ({int(sp * 100)}\\%)"
            match = [r for r in rows
                     if r.get("prune_method") == "wanda"
                     and r.get("calib_data") == calib
                     and r.get("sparsity_ratio_requested") == sp]
            row_specs.append((label, "wanda", (calib, sp), match))

    lines = []
    n_metric_cols = len(interaction_metrics)
    n_delta_cols = n_metric_cols  # one delta per metric
    col_spec = "l" + ("c" * n_metric_cols) + ("c" * n_delta_cols)
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    headers = ["Cal. (sp.)"]
    for _, label, direction in interaction_metrics:
        arrow = "$\\downarrow$" if direction == "down" else "$\\uparrow$"
        headers.append(f"{label} {arrow}")
    for _, label, _ in interaction_metrics:
        headers.append(f"$\\Delta${label}")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for label, _, _, match in row_specs:
        cells = [label]
        values_for_delta = []
        for k, _, _ in interaction_metrics:
            m, s = mean_std([r.get(k) for r in match])
            cells.append(fmt_metric_value(m, s, k))
            values_for_delta.append(m)
        for (k, _, direction), m in zip(interaction_metrics, values_for_delta):
            ref = dense_ref.get(k)
            if m is None or ref is None:
                cells.append("--")
            else:
                # For "down" metrics (ppl), positive delta = degradation.
                # For "up" metrics (acc), negative delta = degradation.
                delta = m - ref if direction == "down" else ref - m
                if "ppl" in k:
                    cells.append(f"{delta:+.2f}")
                else:
                    cells.append(f"{(delta * 100):+.2f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"% Generated by fill_tables.py. $\\Delta$ = degradation vs dense "
                 f"(positive for ppl, accuracy-pp-drop for acc).")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_generic_pivot(rows, out_path, row_var, col_var, metric_keys,
                        filter_kv=None, model_filter=None):
    """Generic pivot: rows by row_var, columns by col_var, cells = mean of metric.

    metric_keys: list of (key, label, direction) tuples.
    filter_kv: dict of key=value filters applied to rows before pivoting.

    Useful for ad-hoc tables not covered by the named generators.
    """
    rows = [r for r in rows if (model_filter is None or r.get("model") == model_filter)]
    if filter_kv:
        for k, v in filter_kv.items():
            rows = [r for r in rows if str(r.get(k)) == str(v)]

    grouped = group_rows(rows, row_var, col_var)
    row_vals = sorted({k[0] for k in grouped})
    col_vals = sorted({k[1] for k in grouped})

    lines = []
    col_spec = "l" + "c" * (len(col_vals) * len(metric_keys))
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    # Two-level header: top row is col_var values; second is metrics.
    top = [""] + [f"\\multicolumn{{{len(metric_keys)}}}{{c}}{{{c}}}" for c in col_vals]
    lines.append(" & ".join(top) + " \\\\")
    sub = [row_var]
    for _ in col_vals:
        sub.extend(label for _, label, _ in metric_keys)
    lines.append(" & ".join(sub) + " \\\\")
    lines.append("\\midrule")

    for rv in row_vals:
        cells = [str(rv)]
        for cv in col_vals:
            cell_rows = grouped.get((rv, cv), [])
            for k, _, _ in metric_keys:
                m, s = mean_std([r.get(k) for r in cell_rows])
                cells.append(fmt_metric_value(m, s, k))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True,
                        help="Path to summary.csv from aggregate.py")
    parser.add_argument("--out", required=True, help="Output directory for .tex fragments.")
    parser.add_argument("--model", default=None,
                        help="If set, restrict to runs of this model.")
    args = parser.parse_args()

    rows = load_summary(args.summary)
    print(f"loaded {len(rows)} rows from {args.summary}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_exp1_calib_domain(rows, out_dir / "exp1_calib_domain_50.tex",
                            sparsity=0.5, model_filter=args.model)
    write_exp1_calib_domain(rows, out_dir / "exp1_calib_domain_70.tex",
                            sparsity=0.7, model_filter=args.model)
    write_exp3_calib_x_sparsity(rows, out_dir / "exp3_calib_x_sparsity.tex",
                                model_filter=args.model)

    print(f"wrote tables to {out_dir}/")
    print("  exp1_calib_domain_50.tex   calibration source x metric, sparsity 50%")
    print("  exp1_calib_domain_70.tex   calibration source x metric, sparsity 70%")
    print("  exp3_calib_x_sparsity.tex  C4/MC4-ko x 50/70 with delta-vs-dense")


if __name__ == "__main__":
    main()
