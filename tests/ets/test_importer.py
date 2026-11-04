import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from gaia.ets.importer import (
    build_ets_home,
    classify_load,
    import_knxdata,
    inventory_stats,
    is_residential,
    load_layouts,
)
from gaia.household.personas import make_household
from gaia.rng import rng_for

FIX = Path(__file__).parent / "fixtures"
SAMPLE = FIX / "sample_KNXdata.json"
ROOT = Path(__file__).resolve().parents[2]


def _names() -> list[str]:
    d = json.loads(SAMPLE.read_text())
    return [d["prjNm"], *(r["name"] for r in d["rooms"]), *(ld["ldNm"] for ld in d["loads"])]


def test_room_mapping():
    lay = import_knxdata(SAMPLE)
    types = Counter(t for t, _ in lay.rooms)
    assert types == {"living": 1, "bedroom": 1, "bathroom": 1, "kitchen": 1, "pooja": 1,
                     "dining": 1, "terrace": 1}
    assert len(lay.project_id) == 12


def test_load_mapping():
    assert classify_load("Dimmer", "Living Cove") == ("light.dimmable", "cove", "cove")
    assert classify_load("Dimmer", "Living Downlights") == ("light.dimmable", "downlight", "main")
    assert classify_load("Tunable", "x") == ("light.tunable", "downlight", "main")
    assert classify_load("Switch", "Toilet Exhaust") == ("fan.exhaust", None, "none")
    assert classify_load("Switch", "Living Fan") == ("fan.ceiling", None, "none")
    assert classify_load("Switch", "Geyser") == ("water.geyser", None, "none")
    assert classify_load("Switch", "Bed Socket") == ("plug.smart", None, "none")
    assert classify_load("Switch", "Terrace Step Light") == ("light.onoff", "step", "night")
    assert classify_load("Curtain", "x") == ("shade.curtain", None, "none")
    assert classify_load("HVAC", "x") == ("climate.vrf_indoor", None, "none")
    assert classify_load("Scene", "x") == ("keypad.scene", None, "none")
    assert classify_load("Relay", "x") is None
    lay = import_knxdata(SAMPLE)
    assert lay.stats["loads_unknown_type"] == 1


def test_no_names_in_output(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "p1").mkdir()
    (tmp_path / "in" / "p1" / "KNXdata.json").write_bytes(SAMPLE.read_bytes())
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "scripts/import_ets.py"), str(tmp_path / "in"), str(out)],
                   check=True, cwd=ROOT, capture_output=True)
    text = "".join(p.read_text() for p in out.glob("*.json"))
    assert (out / "summary.json").exists() and len(list(out.glob("*.json"))) == 2
    assert "CLIENTSECRET" not in text
    for name in _names():
        assert name.strip().lower() not in text.lower()


def test_function_grouped_project_not_residential():
    assert not is_residential(import_knxdata(FIX / "office_KNXdata.json"))
    assert is_residential(import_knxdata(SAMPLE))


def test_ets_archetype_builds_home(tmp_path):
    lay = import_knxdata(SAMPLE)
    (tmp_path / f"{lay.project_id}.json").write_text(lay.to_json())
    load_layouts.cache_clear()
    m, spec = build_ets_home(tmp_path, "h1", "Mumbai", rng_for(1, "home"))
    names = {r.room_id.split(":")[-1] for r in m.rooms}
    assert {"living", "kitchen", "bedroom_1", "bathroom_1", "pooja", "dining", "terrace"} <= names
    assert m.archetype == "ets" and spec.bedrooms == 1 and spec.bathrooms == 1
    for r in m.rooms:
        mains = [ld for ld in m.loads_in(r.room_id) if ld.is_main_light]
        assert len(mains) <= 1
    assert len({ld.load_id for ld in m.loads}) == len(m.loads)
    assert any(ld.load_class == "media.tv" for ld in m.loads)  # inventory adds non-KNX loads
    assert make_household(m, spec, rng_for(1, "household")).members
    assert inventory_stats([lay])["bedroom"]["shade.curtain"] == 1.0


def test_missing_ets_dir_is_clear():
    with pytest.raises(ValueError, match="ets_dir"):
        build_ets_home(None, "h1", "Mumbai", rng_for(1, "home"))


def test_ets_config_and_simulation(tmp_path):
    from gaia.config import RunConfig, home_spec
    from gaia.engine.sim import simulate_home

    base = dict(world_version="t", seed=3, homes=1, days=2, start_date="2026-11-02",
                cities=["hyderabad"], archetypes=["ets"], out_dir=tmp_path / "o")
    with pytest.raises(ValueError, match="ets_dir"):
        RunConfig(**base)
    lay = import_knxdata(SAMPLE)
    (tmp_path / f"{lay.project_id}.json").write_text(lay.to_json())
    load_layouts.cache_clear()
    res = simulate_home(home_spec(RunConfig(**base, ets_dir=tmp_path), 0))
    assert res.manifest.archetype == "ets" and res.events


def test_null_names_and_kitchen_wash(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"loads": [{"ldNm": None, "ldTyp": "Dimmer"}],
                             "rooms": [{"name": None, "loads": [1]}]}))
    assert import_knxdata(p).stats["loads_unmapped_room"] == 1
    from gaia.ets.importer import room_type_for
    assert room_type_for("Kitchen Wash Area") == "kitchen"
    assert room_type_for("Guest Washroom") == "bathroom"
