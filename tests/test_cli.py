import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent


def _write_cfg(path: Path, out_dir: Path, homes: int = 2, days: int = 2, workers: int | None = None) -> Path:
    text = (
        f"world_version: g1\nseed: 2026\nhomes: {homes}\ndays: {days}\nstart_date: 2026-11-02\n"
        f"cities: [hyderabad, bengaluru]\nout_dir: {out_dir}\n"
    )
    if workers is not None:
        text += f"workers: {workers}\n"
    path.write_text(text, encoding="utf-8")
    return path


def _generate(cfg: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "gaia", "generate", str(cfg)],
                          cwd=REPO, capture_output=True, text=True, timeout=600)


def _digests(out_dir: Path) -> dict[str, str]:
    return {str(p.relative_to(out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out_dir.rglob("events.parquet"))}


def test_cli_smoke(tmp_path):
    out = tmp_path / "out"
    proc = _generate(_write_cfg(tmp_path / "run.yaml", out))
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["homes"] == 2 and summary["events"] > 0
    shards = sorted(out.glob("world=g1/split=train/home=*"))
    assert len(shards) == 2
    for shard in shards:
        assert pq.read_table(shard / "events.parquet").num_rows > 0


def test_run_twice_identical(tmp_path):
    cfg = _write_cfg(tmp_path / "run.yaml", tmp_path / "a", homes=3, days=2)
    assert _generate(cfg).returncode == 0
    first = _digests(tmp_path / "a")
    cfg_b = _write_cfg(tmp_path / "run_b.yaml", tmp_path / "b", homes=3, days=2)
    assert _generate(cfg_b).returncode == 0
    second = _digests(tmp_path / "b")
    assert first and first == second


def _all_file_digests(out_dir: Path) -> dict[str, str]:
    return {str(p.relative_to(out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out_dir.rglob("*")) if p.is_file()}


def test_worker_count_does_not_change_any_shard_file(tmp_path):
    one = _write_cfg(tmp_path / "w1.yaml", tmp_path / "one", homes=4, days=2, workers=1)
    two = _write_cfg(tmp_path / "w2.yaml", tmp_path / "two", homes=4, days=2, workers=2)
    assert _generate(one).returncode == 0
    proc = _generate(two)
    assert proc.returncode == 0, proc.stderr
    digests_one = _all_file_digests(tmp_path / "one")
    assert len(list((tmp_path / "one").glob("world=g1/split=train/home=*"))) == 4
    assert {p.rsplit("/", 1)[-1] for p in digests_one} >= {
        "events.parquet", "context.parquet", "gt_cause.parquet", "gt_waste.parquet",
        "gt_presence.parquet", "manifest.json"}
    assert digests_one == _all_file_digests(tmp_path / "two")
