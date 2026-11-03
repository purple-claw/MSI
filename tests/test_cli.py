import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent


def _write_cfg(path: Path, out_dir: Path, homes: int = 2, days: int = 2) -> Path:
    path.write_text(
        f"world_version: g1\nseed: 2026\nhomes: {homes}\ndays: {days}\nstart_date: 2026-11-02\n"
        f"cities: [hyderabad, bengaluru]\nout_dir: {out_dir}\n",
        encoding="utf-8",
    )
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
