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
    python orchestrator.py --config configs/phase3_main.yaml --max_parallel 8
    python orchestrator.py --config configs/phase3_main.yaml --max_parallel 4 \
        --gpu_ids 0,1,2,3

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
      # Optional: env vars merged into the subprocess environment.
      env:
        CUDA_VISIBLE_DEVICES: "0"
    runs:
      - prune_method: wanda
        calib_data: c4
        sparsity_ratio: 0.5
        seeds: [0, 1, 2]                 # expands into one cell per seed
        save_template: out/phase3/wanda_c4_50_seed{seed}/

Boolean True becomes a bare CLI flag (--eval_zero_shot); False is omitted.
None values are omitted. The 'env' key is NOT emitted as a CLI flag --
it merges into the subprocess environment instead.

Parallelism:
    --max_parallel N runs up to N cells concurrently. Each cell is
    assigned a GPU from --gpu_ids (default 0..N-1) via CUDA_VISIBLE_DEVICES.
    A cell-level env override (env: {CUDA_VISIBLE_DEVICES: ...}) takes
    precedence over the orchestrator's auto-assignment.

    On an 8x A6000 machine, --max_parallel 8 cuts Phase 3 wall time
    from ~32h sequential to ~4-5h. Each Wanda run uses ~16 GB of one
    48 GB GPU, so multi-cells-per-GPU is also viable; e.g. --max_parallel
    16 --gpu_ids 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 packs 2 per GPU.

Why not bash for-loops:
    32+ cells in Phase 3, each with 8-10 distinct CLI args. Manually
    edited bash commands invite typos and silent wasted compute. The
    YAML config doubles as a version-controlled record of what was run.
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

    Each cell is a flat dict where every key is either a main.py CLI arg
    or the special 'env' key (subprocess environment overrides).
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
            # Merge env separately so per-run env keys layer onto defaults
            # without replacing the whole dict.
            cell_env = dict((defaults.get("env") or {}))
            cell_env.update(run.get("env") or {})
            cell = {k: v for k, v in defaults.items() if k != "env"}
            cell.update({k: v for k, v in run.items()
                         if k not in ("seeds", "save_template", "save", "env")})
            cell["seed"] = seed
            if cell_env:
                cell["env"] = cell_env
            if save_template:
                # seed is already in cell at this point; pass cell scalars only.
                cell["save"] = save_template.format(**{
                    k: v for k, v in cell.items() if isinstance(v, (str, int, float))
                })
            cells.append(cell)
    return cells


def build_command(cell, main_script, python_bin):
    """Convert a cell dict to a `python main.py ...` command list.

    Booleans True -> bare flag; False -> omitted. None values -> omitted.
    The 'env' key is skipped (handled separately as a subprocess env).
    Everything else becomes `--key value` (with str(value)).
    """
    cmd = [python_bin, main_script]
    for key, val in cell.items():
        if key == "env":
            continue
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(val)])
    return cmd


def build_env(cell_env_overrides, gpu_id):
    """Build the subprocess environment for one cell.

    Starts from os.environ.copy(). Applies cell-level overrides. If
    CUDA_VISIBLE_DEVICES is not set by the cell AND gpu_id is given,
    set it to gpu_id. Cell-level overrides always take precedence.
    """
    env = os.environ.copy()
    overrides = cell_env_overrides or {}
    cell_sets_cuda = "CUDA_VISIBLE_DEVICES" in overrides
    for k, v in overrides.items():
        env[str(k)] = str(v)
    if gpu_id is not None and not cell_sets_cuda:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def cell_complete(save_dir):
    """A cell is 'complete' if its manifest.json exists."""
    if not save_dir:
        return False
    return Path(save_dir).joinpath("manifest.json").exists()


def run_cells(cells, main_script, python_bin, max_parallel, gpu_ids,
              dry_run=False, resume=False, poll_interval=2.0):
    """Execute cells with bounded parallelism.

    cells: list of cell dicts (from expand_runs).
    max_parallel: max concurrent processes; 1 = sequential.
    gpu_ids: list of GPU IDs for round-robin (length >= max_parallel).
    Returns: list of summary dicts in original cell order.
    """
    if gpu_ids is None or not gpu_ids:
        gpu_ids = list(range(max_parallel))
    if len(gpu_ids) < max_parallel:
        raise ValueError(f"need at least max_parallel={max_parallel} gpu_ids; "
                         f"got {gpu_ids}")

    summary = [None] * len(cells)
    in_flight = {}  # idx -> {proc, slot, gpu, start, save_dir, logf}
    used_slots = set()
    next_idx = 0

    def alloc_slot():
        for s in range(max_parallel):
            if s not in used_slots:
                return s
        return None

    def launch(idx):
        cell = cells[idx]
        save_dir = cell.get("save")
        cmd = build_command(cell, main_script, python_bin)

        if resume and cell_complete(save_dir):
            summary[idx] = {"idx": idx, "save_dir": save_dir,
                            "status": "SKIP", "elapsed": 0, "gpu": None}
            print(f"  [{idx+1}/{len(cells)}] SKIP (manifest.json exists) {save_dir}")
            return None

        slot = alloc_slot()
        gpu = gpu_ids[slot]
        env = build_env(cell.get("env"), gpu)

        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        cuda_str = env.get("CUDA_VISIBLE_DEVICES", "<unset>")

        if dry_run:
            summary[idx] = {"idx": idx, "save_dir": save_dir,
                            "status": "DRY_RUN", "elapsed": 0,
                            "gpu": cuda_str, "cmd": cmd_str}
            print(f"  [{idx+1}/{len(cells)}] DRY CUDA={cuda_str}: {cmd_str}")
            return None

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(save_dir) / "orchestrator.log"
        logf = open(log_path, "w")
        logf.write(f"# command: {cmd_str}\n")
        logf.write(f"# CUDA_VISIBLE_DEVICES={cuda_str}\n")
        logf.write(f"# slot={slot}\n\n")
        logf.flush()

        used_slots.add(slot)
        print(f"  [{idx+1}/{len(cells)}] launch CUDA={cuda_str} slot={slot}: {save_dir}")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        return {"proc": proc, "slot": slot, "gpu": cuda_str,
                "start": time.time(), "save_dir": save_dir, "logf": logf}

    while next_idx < len(cells) or in_flight:
        # Fill empty slots
        while len(in_flight) < max_parallel and next_idx < len(cells):
            idx = next_idx
            next_idx += 1
            launched = launch(idx)
            if launched is not None:
                in_flight[idx] = launched

        if not in_flight:
            break

        time.sleep(poll_interval)

        # Reap finished
        done_idxs = []
        for idx, meta in in_flight.items():
            rc = meta["proc"].poll()
            if rc is None:
                continue
            elapsed = time.time() - meta["start"]
            meta["logf"].close()
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            summary[idx] = {"idx": idx, "save_dir": meta["save_dir"],
                            "status": status, "elapsed": elapsed,
                            "gpu": meta["gpu"]}
            print(f"  [{idx+1}/{len(cells)}] {status:<12} elapsed={elapsed:>7.1f}s "
                  f"GPU={meta['gpu']} {meta['save_dir']}")
            done_idxs.append(idx)
        for idx in done_idxs:
            used_slots.discard(in_flight[idx]["slot"])
            del in_flight[idx]

    return summary


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
    parser.add_argument("--max_parallel", type=int, default=1,
                        help="Run up to N cells concurrently. Each is assigned "
                             "one GPU from --gpu_ids via CUDA_VISIBLE_DEVICES. "
                             "Default 1 (sequential).")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated list of GPU IDs to round-robin "
                             "across. Default: 0..max_parallel-1. To pack "
                             "multiple cells per GPU use e.g. --gpu_ids 0,0,1,1 "
                             "with --max_parallel 4.")
    parser.add_argument("--poll_interval", type=float, default=2.0,
                        help="Seconds between polls of in-flight processes.")
    args = parser.parse_args()

    if args.max_parallel < 1:
        parser.error("--max_parallel must be >= 1")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    name = config.get("name", Path(args.config).stem)
    python_bin = config.get("python_bin", sys.executable)
    main_script = config.get("main_script", "main.py")

    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.max_parallel))

    cells = expand_runs(config)
    if args.start_from:
        cells = cells[args.start_from:]
    if args.limit is not None:
        cells = cells[:args.limit]

    print(f"orchestrator '{name}': {len(cells)} cells to consider")
    print(f"main script:  {main_script}")
    print(f"python bin:   {python_bin}")
    print(f"max_parallel: {args.max_parallel}")
    print(f"gpu_ids:      {gpu_ids}")
    if args.dry_run:
        print("DRY RUN -- nothing will execute")

    summary = run_cells(
        cells, main_script, python_bin,
        max_parallel=args.max_parallel,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        resume=args.resume,
        poll_interval=args.poll_interval,
    )

    print(f"\n===== orchestrator '{name}' summary =====")
    for s in summary:
        if s is None:
            continue
        gpu = s.get("gpu", "-")
        print(f"  [{s['idx']+1:>3}] {s['status']:<14}  elapsed={s['elapsed']:>8.1f}s "
              f"GPU={gpu}  {s['save_dir']}")

    ok = sum(1 for s in summary if s and s["status"] == "OK")
    skip = sum(1 for s in summary if s and s["status"] == "SKIP")
    fail = sum(1 for s in summary if s and ("FAIL" in s["status"] or s["status"] == "EXCEPTION"))
    dry = sum(1 for s in summary if s and s["status"] == "DRY_RUN")
    total = sum(1 for s in summary if s is not None)
    print(f"\n  OK: {ok}/{total}   SKIP: {skip}   FAIL: {fail}   DRY_RUN: {dry}")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
