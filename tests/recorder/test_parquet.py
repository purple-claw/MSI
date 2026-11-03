import hashlib
import json
from datetime import date

import pyarrow.parquet as pq

from gaia.engine.sim import HomeSpec, simulate_home
from gaia.recorder.parquet import shard_dir, write_home


def _result(days=2):
    spec = HomeSpec(home_id="h1", home_seed=7, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=days, festivals=frozenset())
    return simulate_home(spec)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_roundtrip_event_count(tmp_path):
    result = _result()
    shard = write_home(result, tmp_path, "g1", "train")
    assert shard == shard_dir(tmp_path, "g1", "train", "h1")
    assert pq.read_table(shard / "events.parquet").num_rows == len(result.events)
    assert pq.read_table(shard / "context.parquet").num_rows == len(result.context)
    assert pq.read_table(shard / "gt_cause.parquet").num_rows == len(result.gt_cause)
    assert pq.read_table(shard / "gt_waste.parquet").num_rows == len(result.gt_waste)
    manifest = json.loads((shard / "manifest.json").read_text())
    assert manifest["world_version"] == "g1"
    assert [m["role"] for m in manifest["household"]] == [m.persona.role for m in result.household.members]


def test_rewrite_overwrites_not_appends(tmp_path):
    result = _result()
    write_home(result, tmp_path, "g1", "train")
    shard = write_home(result, tmp_path, "g1", "train")
    assert pq.read_table(shard / "events.parquet").num_rows == len(result.events)
    leftovers = [p.name for p in shard.parent.iterdir() if ".tmp-" in p.name or ".old-" in p.name]
    assert leftovers == []


def test_bytes_stable(tmp_path):
    result = _result()
    a = write_home(result, tmp_path / "a", "g1", "train") / "events.parquet"
    b = write_home(result, tmp_path / "b", "g1", "train") / "events.parquet"
    assert _sha(a) == _sha(b)
