"""Write one home's simulation result as a Parquet shard, atomically.

A shard is a directory holding events, context, ground-truth labels and a manifest.
It is built in a temporary sibling directory and swapped into place, so a rewrite
replaces the old shard and leaves nothing behind.
"""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gaia.engine.sim import SimResult
from gaia.schema.context import (
    CONTEXT_ARROW_SCHEMA,
    GT_CAUSE_ARROW_SCHEMA,
    GT_PRESENCE_ARROW_SCHEMA,
    GT_WASTE_ARROW_SCHEMA,
    LABEL_ARROW_SCHEMA,
)
from gaia.schema.events import events_to_table


def shard_dir(out_dir: Path, world_version: str, split: str, home_id: str) -> Path:
    return out_dir / f"world={world_version}" / f"split={split}" / f"home={home_id}"


def _manifest_json(result: SimResult, world_version: str) -> str:
    payload = json.loads(result.manifest.to_json())
    payload["household"] = [
        {
            "person_id": m.person_id,
            "role": m.persona.role,
            "bedroom": m.bedroom,
            "bathroom": m.bathroom,
        }
        for m in result.household.members
    ]
    payload["world_version"] = world_version
    payload["home_seed"] = result.spec.home_seed
    payload["start_date"] = result.spec.start_date.isoformat()
    payload["days"] = result.spec.days
    payload["festivals"] = sorted(d.isoformat() for d in result.spec.festivals)
    return json.dumps(payload, sort_keys=True, indent=None, separators=(",", ":"))


def _write_files(result: SimResult, directory: Path, world_version: str) -> None:
    tables = {
        "events.parquet": events_to_table(result.events),
        "context.parquet": pa.Table.from_pylist(
            [asdict(r) for r in result.context], schema=CONTEXT_ARROW_SCHEMA
        ),
        "gt_cause.parquet": pa.Table.from_pylist(
            [asdict(g) for g in result.gt_cause], schema=GT_CAUSE_ARROW_SCHEMA
        ),
        "labels.parquet": pa.Table.from_pylist(
            [asdict(lb) for lb in result.labels], schema=LABEL_ARROW_SCHEMA
        ),
        "gt_waste.parquet": pa.Table.from_pylist(
            [asdict(w) for w in result.gt_waste], schema=GT_WASTE_ARROW_SCHEMA
        ),
        "gt_presence.parquet": pa.Table.from_pylist(
            [asdict(p) for p in result.gt_presence], schema=GT_PRESENCE_ARROW_SCHEMA
        ),
    }
    for name, table in tables.items():
        pq.write_table(table, directory / name, compression="zstd")
    (directory / "manifest.json").write_text(_manifest_json(result, world_version), encoding="utf-8")


def write_home(result: SimResult, out_dir: Path, world_version: str, split: str) -> Path:
    """Write the shard for one home and return its directory.

    An existing shard for the same home is replaced, not appended to.
    """
    target = shard_dir(out_dir, world_version, split, result.manifest.home_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        _write_files(result, tmp, world_version)
        if target.exists():
            old = target.with_name(f"{target.name}.old-{os.getpid()}")
            os.replace(target, old)
            try:
                os.replace(tmp, target)
            except BaseException:
                os.replace(old, target)
                raise
            shutil.rmtree(old)
        else:
            os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target
