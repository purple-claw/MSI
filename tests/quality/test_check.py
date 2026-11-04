import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gaia.cli import generate
from gaia.config import RunConfig
from gaia.home.archetypes import ARCHETYPE_SPECS
from gaia.quality.check import check_dataset

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("qc") / "out"
    cfg = RunConfig(world_version="qc", seed=11, homes=6, days=7, start_date="2026-11-02",
                    cities=["hyderabad", "bengaluru"], archetypes=sorted(ARCHETYPE_SPECS),
                    out_dir=out, workers=1)
    generate(cfg)
    return out


def test_clean_world_passes(world):
    report = check_dataset(world)
    assert report["violations"] and not any(report["violations"].values()), report["examples"]
    assert report["ok"] and report["stats"]["homes"] == 6
    cov = report["stats"]["label_coverage"]
    assert cov["intent"] >= 0.99 and cov["device_action"] >= 0.99


def test_detects_injected_violation(world, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(world, copy)
    shard = next(copy.rglob("labels.parquet"))
    table = pq.read_table(shard)
    pq.write_table(table.slice(1), shard)  # drop the first label row
    report = check_dataset(copy)
    assert report["violations"]["command_label_count"] >= 1 and not report["ok"]


def test_cli_check_exit_code(world, tmp_path):
    run = [sys.executable, "-m", "gaia", "check"]
    ok = subprocess.run([*run, str(world)], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert ok.returncode == 0, ok.stdout[-2000:]
    assert json.loads(ok.stdout)["ok"] is True
    bad = tmp_path / "bad"
    shutil.copytree(world, bad)
    shard = next(bad.rglob("labels.parquet"))
    pq.write_table(pq.read_table(shard).slice(1), shard)
    fail = subprocess.run([*run, str(bad)], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert fail.returncode == 1


def test_empty_root_is_a_violation(tmp_path):
    report = check_dataset(tmp_path)
    assert report["violations"]["no_homes"] == 1 and not report["ok"]
