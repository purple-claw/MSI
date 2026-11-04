from datetime import date

import pydantic
import pytest

from gaia.config import RunConfig, home_spec, load_config
from gaia.home.archetypes import APARTMENTS


def _cfg(**overrides):
    base = dict(world_version="g1", seed=2026, homes=8, days=30, start_date=date(2026, 11, 2),
                cities=["hyderabad", "bengaluru"], out_dir="data/test")
    base.update(overrides)
    return RunConfig(**base)


def test_rejects_bad_config():
    with pytest.raises(pydantic.ValidationError):
        _cfg(homes=0)
    with pytest.raises(pydantic.ValidationError):
        _cfg(days=-1)
    with pytest.raises(pydantic.ValidationError):
        _cfg(cities=["atlantis"])


def test_home_spec_independent_of_count():
    small = _cfg(homes=8)
    large = _cfg(homes=800)
    assert home_spec(small, 7) == home_spec(large, 7)


def test_archetypes_default_to_all_apartments():
    assert _cfg().archetypes == list(APARTMENTS)


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("world_version: g1\nseed: 1\nhomes: 3\ndays: 2\nstart_date: 2026-11-02\n"
                    "cities: [chennai]\nout_dir: out\n", encoding="utf-8")
    cfg = load_config(path)
    assert (cfg.homes, cfg.days, cfg.cities, cfg.split, cfg.workers) == (3, 2, ["chennai"], "train", 0)
