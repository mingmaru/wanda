"""
Multi-cell experiment orchestrator for the Wanda stress test.

Reads a YAML config describing a matrix of (model, calibration, sparsity,
seed) cells. For each cell, builds and runs an appropriate `main.py`
invocation. Per-cell stdout/stderr go to the cell's save directory;
orchestrator emits a top-level summary at the end.

Usage:
    python orchestrator.py --config configs/phase3_main.yaml --dry_run
    python orchestrator.py --config configs/phase3_main.yaml
    python orchestrator.py --config configs/phase3_main.yaml --resume
    python orchestrator.py --config configs/phase3_main.yaml --limit 3

Config schema (YAML):
    name: <human-readable label for the run set>
    main_script: main.py                # optional, defaults to main.py
    python_bin: /path/to/python          # optional, defaults to sys.executable
    defaults:
      # Any main.py CLI arg can go here; applied to every run.
      model: meta-llama/Meta-Llama-3-8B
      sparsity_type: unstructured
      nsamples: 128
      eval_zero_shot: true
      eval_tasks: headline
      eval_korean_ppl: true
    runs:
      - prune_method: wanda
        calib_data: c4
        sparsity_ratio: 0.5
        seeds: [0, 1, 2]                 # expands into one cell per seed
        save_template: out/phase3/wanda_c4_50_seed{seed}/
      ...

Boolean True becomes a bare CLI flag (--eval_zero_shot); False is omitted.
None values are omitted. Everything else becomes `--key value`.

Why not bash for-loops:
    32+ cells in Phase 3, each with 8-10 distinct CLI args. Manually
    editing that many bash commands invites typos and silent wasted
    compute. A config-driven runner with --resume support recovers from
    partial failures cleanly.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def expand_runs(config):
    """Expand each run spec into one fully-resolved cell per seed.

    Each cell is a flat dict where every key is a main.py CLI arg.
    """
    cells = []
    defaults = config.get("defaults", {}) or {}
    for run in config["runs"]:
        seeds = run.get("seeds")
        if seeds is None:
            seeds = [run.get("seed", 0)]
        if not isinstance(seeds, list):
            seeds = [seeds]
        save_template = run.get("save_template") or run.get("save")
        for seed in seeds:
            cell = dict(defaults)
            cell.update({k: v for k, v in run.items()
                         if k not in ("seeds", "save_template", "save")})
            cell["seed"] = seed
            if save_template:
                cell["save"] = save_template.format(seed=seed, **{
                    k: v for k, v in cell.items() if isinstance(v, (str, int, float))
                })
            cells.append(cell)
    return cells


def build_command(cell, main_script, python_bin):
    """Convert a cell dict to a `python main.py ...` command list.

    Booleans True -> bare flag; False -> omitted. None values -> omitted.
    Everything else becomes `--key value` (with str(value)).
    """
    cmd = [python_bin, main_script]
    for key, val in cell.items():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(val)])
    return cmd


def cell_complete(save_dir):
    """A cell is 'complete' if its manifest.json exists."""
    if not save_dir:
        return False
    return Path(save_dir).joinpath("manifest.json").exists()


def run_cell(cmd, save_dir, log_path):
    """Run the command, redirecting stdout+stderr to log_path."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"# orchestrator command:\n# {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose save dir already has manifest.json.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N cells (debugging).")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Skip the first N cells (resume from partial failures).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    name = config.get("name", Path(args.config).stem)
    python_bin = config.get("python_bin", sys.executable)
    main_script = config.get("main_script", "main.py")

    cells = expand_runs(config)
    if args.start_from:
        cells = cells[args.start_from:]
    if args.limit is not None:
        cells = cells[:args.limit]

    print(f"orchestrator '{name}': {len(cells)} cells to consider")
    print(f"main script: {main_script}")
    print(f"python bin:  {python_bin}")
    if args.dry_run:
        print("DRY RUN -- nothing will execute")

    summary = []
    for i, cell in enumerate(cells):
        save_dir = cell.get("save")
        cmd = build_command(cell, main_script, python_bin)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        print(f"\n[{i+1}/{len(cells)}] save={save_dir}")

        if args.resume and cell_complete(save_dir):
            print(f"  -> SKIP (manifest.json already in {save_dir})")
            summary.append((i, save_dir, "SKIP", 0))
            continue

        if args.dry_run:
            print(f"  -> {cmd_str}")
            summary.append((i, save_dir, "DRY_RUN", 0))
            continue

        log_path = (Path(save_dir) / "orchestrator.log") if save_dir else Path(f"orchestrator_cell_{i}.log")
        print(f"  -> running: {cmd_str}")
        print(f"  -> log:     {log_path}")
        try:
            rc, elapsed = run_cell(cmd, save_dir, log_path)
        except Exception as e:
            print(f"  -> EXCEPTION: {type(e).__name__}: {e}")
            summary.append((i, save_dir, "EXCEPTION", 0))
            continue
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  -> {status}  elapsed={elapsed:.1f}s")
        summary.append((i, save_dir, status, elapsed))

    print(f"\n===== orchestrator '{name}' summary =====")
    for i, sd, status, elapsed in summary:
        print(f"  [{i+1:>3}] {status:<14}  {elapsed:>8.1f}s  {sd}")

    ok = sum(1 for _, _, s, _ in summary if s == "OK")
    skip = sum(1 for _, _, s, _ in summary if s == "SKIP")
    fail = sum(1 for _, _, s, _ in summary if "FAIL" in s or s == "EXCEPTION")
    dry = sum(1 for _, _, s, _ in summary if s == "DRY_RUN")
    total = len(summary)
    print(f"\n  OK: {ok}/{total}   SKIP: {skip}   FAIL: {fail}   DRY_RUN: {dry}")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
