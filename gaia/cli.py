"""Command line entry point: `python -m gaia generate <config.yaml>` and `python -m gaia check <root>`."""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from gaia.config import RunConfig, home_spec, load_config
from gaia.engine.sim import simulate_home
from gaia.quality.check import check_dataset
from gaia.recorder.parquet import write_home


def _run_one(cfg: RunConfig, index: int) -> int:
    result = simulate_home(home_spec(cfg, index))
    write_home(result, cfg.out_dir, cfg.world_version, cfg.split)
    return len(result.events)


def generate(cfg: RunConfig) -> dict:
    """Simulate and record every home. workers=1 runs inline; otherwise a process pool."""
    workers = cfg.workers or os.cpu_count() or 1
    start = time.perf_counter()
    if workers == 1:
        counts = [_run_one(cfg, i) for i in range(cfg.homes)]
    else:
        chunksize = max(1, cfg.homes // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            counts = list(pool.map(partial(_run_one, cfg), range(cfg.homes), chunksize=chunksize))
    seconds = time.perf_counter() - start
    home_years = cfg.homes * cfg.days / 365
    return {
        "homes": cfg.homes,
        "events": sum(counts),
        "seconds": round(seconds, 3),
        "home_years": home_years,
        "home_years_per_hour": round(home_years / seconds * 3600, 1) if seconds > 0 else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gaia")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate home shards from a run config")
    gen.add_argument("config", type=Path, help="path to a run config YAML file")
    chk = sub.add_parser("check", help="check a generated world; exit 1 on any violation")
    chk.add_argument("root", type=Path, help="output directory of a run")
    args = parser.parse_args(argv)
    if args.command == "generate":
        cfg = load_config(args.config)
        summary = generate(cfg)
        report = check_dataset(cfg.out_dir)
        (Path(cfg.out_dir) / "check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        summary["check_ok"] = report["ok"]
        print(json.dumps(summary))
        return 0 if report["ok"] else 1
    report = check_dataset(args.root)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1
