"""Dataset checker: hard violations (must be 0) and quality stats over a generated world.

check_dataset(root) walks every home shard (a directory holding events.parquet) under root.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from gaia.labels.model import _ENUM_FIELDS, ENUMS
from gaia.registry.registry import REGISTRY

MS_PER_MIN = 60_000
MOTION_SLACK_MS = MS_PER_MIN  # sensor timestamps may sit at most a minute off the presence edge
VIOLATIONS = (
    "duplicate_event_id", "non_monotonic_state", "state_without_command",
    "command_label_count", "label_missing_event", "label_enum", "value_out_of_range",
    "waste_on_off_load", "presence_overlap", "motion_from_sleeper", "empty_home",
)


def _shards(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).rglob("events.parquet"))


def _rows(shard: Path, name: str) -> list[dict]:
    path = shard / name
    return pq.read_table(path).to_pylist() if path.exists() else []


def _in_range(load_class: str | None, capability: str | None, value: str | None) -> bool:
    if load_class is None or capability is None or value is None:
        return True
    cls = REGISTRY.get(load_class)
    if cls is None:
        return False
    cap = next((c for c in cls.capabilities if c.name == capability), None)
    if cap is None:
        return False
    if cap.values:
        return value in cap.values
    try:
        x = float(value)
    except ValueError:
        return False
    return (cap.lo is None or x >= cap.lo) and (cap.hi is None or x <= cap.hi)


def _is_on(cap: str, value: str) -> bool:
    try:
        return float(value) > 0
    except ValueError:
        return value not in ("off", "stopped", "closed")


def check_home(shard: Path, out: dict, acc: dict) -> None:
    v = out["violations"]
    events = _rows(shard, "events.parquet")
    labels = _rows(shard, "labels.parquet")
    waste = _rows(shard, "gt_waste.parquet")
    presence = _rows(shard, "gt_presence.parquet")
    manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
    room_type = {r["room_id"]: r["room_type"] for r in manifest.get("rooms", [])}
    load_room = {ld["load_id"]: ld["room_id"] for ld in manifest.get("loads", [])}
    days = max(1, int(manifest.get("days", 0)) or 1)

    def bad(kind: str, detail: str) -> None:
        v[kind] += 1
        if len(out["examples"]) < 20:
            out["examples"].append(f"{shard.name}: {kind}: {detail}")

    if manifest.get("household") and not events:
        bad("empty_home", "no events")

    by_id: dict[str, dict] = {}
    for e in events:
        if e["event_id"] in by_id:
            bad("duplicate_event_id", e["event_id"])
        by_id[e["event_id"]] = e

    last_ts: dict[str, int] = {}
    last_val: dict[tuple[str, str], str] = {}
    on_timeline: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for e in events:
        if e["kind"] == "state":
            lid = e["load_id"]
            if e["ts_utc"] < last_ts.get(lid, -1):
                bad("non_monotonic_state", e["event_id"])
            last_ts[lid] = e["ts_utc"]
            key = (lid, e["capability"])
            if key in last_val and e["prev_value"] is not None and e["prev_value"] != last_val[key]:
                bad("non_monotonic_state", f"{e['event_id']} prev {e['prev_value']} != {last_val[key]}")
            last_val[key] = e["value"]
            ref = by_id.get(e["ref_id"]) if e["ref_id"] else None
            if ref is None or ref["kind"] != "command":
                bad("state_without_command", e["event_id"])
            if e["capability"] in ("power", "position"):
                on_timeline[lid].append((e["ts_utc"], _is_on(e["capability"], e["value"])))
        if e["kind"] in ("command", "state", "sensor") and not _in_range(
                e["load_class"], e["capability"], e["value"]):
            bad("value_out_of_range", f"{e['load_class']}.{e['capability']}={e['value']}")

    per_event = Counter(lb["event_id"] for lb in labels)
    for e in events:
        if e["kind"] == "command" and per_event[e["event_id"]] != 1:
            bad("command_label_count", f"{e['event_id']} has {per_event[e['event_id']]}")
    for lb in labels:
        if lb["event_id"] not in by_id:
            bad("label_missing_event", lb["event_id"])
        for f in _ENUM_FIELDS:
            if lb[f] not in ENUMS[f]:
                bad("label_enum", f"{f}={lb[f]}")

    for w in waste:
        tl = on_timeline.get(w["load_id"], [])
        state_at_start = None
        for ts, on in tl:
            if ts <= w["start_ts_utc"]:
                state_at_start = on
            elif w["end_ts_utc"] is None or ts < w["end_ts_utc"]:
                if not on:
                    bad("waste_on_off_load", f"{w['load_id']} off inside interval")
                    break
        if state_at_start is False:
            bad("waste_on_off_load", f"{w['load_id']} off at start")

    by_person: dict[str, list[dict]] = defaultdict(list)
    for p in presence:
        by_person[p["person_id"]].append(p)
    for spans in by_person.values():
        spans.sort(key=lambda p: p["start_ts_utc"])
        for a, b in zip(spans, spans[1:]):
            if b["start_ts_utc"] < a["end_ts_utc"]:
                bad("presence_overlap", a["person_id"])

    awake_in: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for p in presence:
        if p["room_id"] and not p["asleep"]:
            awake_in[p["room_id"]].append((p["start_ts_utc"], p["end_ts_utc"]))
    for e in events:
        if e["kind"] == "sensor" and e["load_class"] == "sensor.motion" and _is_on("", e["value"] or "0"):
            ts = e["ts_utc"]
            if not any(s - MOTION_SLACK_MS <= ts <= t + MOTION_SLACK_MS for s, t in awake_in[e["room_id"]]):
                bad("motion_from_sleeper", e["event_id"])

    # stats accumulators
    acc["home_days"] += days
    acc["events"] += len(events)
    for e in events:
        if e["load_class"]:
            acc["classes"][e["load_class"]] += 1
        acc["domain"][e["domain"]] += 1
        if e["protocol"]:
            acc["protocol"][e["protocol"]] += 1
    for lb in labels:  # coverage over command labels: sensor labels carry device_action none
        ev = by_id.get(lb["event_id"])
        if ev is None or ev["kind"] != "command":
            continue
        acc["labels"] += 1
        for f in ("activity", "intent", "trigger", "device_action"):
            if lb[f] not in ("unknown", "none"):
                acc["covered"][f] += 1
        acc["intent"][lb["intent"]] += 1
    for w in waste:
        if w["end_ts_utc"] is None:
            continue
        minutes = (w["end_ts_utc"] - w["start_ts_utc"]) / MS_PER_MIN
        acc["waste_min"][w["load_class"]] += minutes
        acc["waste_by_room"][room_type.get(load_room.get(w["load_id"]), "?")].append(minutes)


def check_dataset(root: Path) -> dict:
    out = {"violations": dict.fromkeys(VIOLATIONS, 0), "examples": []}
    out["violations"] = Counter(out["violations"])
    acc = {"home_days": 0, "events": 0, "labels": 0, "classes": Counter(), "domain": Counter(),
           "protocol": Counter(), "covered": Counter(), "intent": Counter(),
           "waste_min": Counter(), "waste_by_room": defaultdict(list)}
    shards = _shards(root)
    for shard in shards:
        check_home(shard, out, acc)

    def share(c: Counter) -> dict[str, float]:
        n = sum(c.values()) or 1
        return {k: round(x / n, 4) for k, x in c.most_common()}

    hd = acc["home_days"] or 1
    out["violations"] = {k: out["violations"][k] for k in VIOLATIONS}
    out["ok"] = not any(out["violations"].values())
    out["stats"] = {
        "homes": len(shards),
        "home_days": acc["home_days"],
        "events_per_home_day": round(acc["events"] / hd, 1),
        "distinct_classes": len(acc["classes"]),
        "class_share": share(acc["classes"]),
        "domain_share": share(acc["domain"]),
        "protocol_share": share(acc["protocol"]),
        "label_coverage": {f: round(acc["covered"][f] / (acc["labels"] or 1), 4)
                           for f in ("activity", "intent", "trigger", "device_action")},
        "intent_share": share(acc["intent"]),
        "waste_min_per_home_day": {k: round(x / hd, 2) for k, x in acc["waste_min"].most_common()},
        "median_waste_min_by_room": {k: round(statistics.median(x), 1)
                                     for k, x in sorted(acc["waste_by_room"].items())},
    }
    return out
