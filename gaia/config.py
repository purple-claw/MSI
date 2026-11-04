"""Run config: what to generate, and the deterministic per-home spec derived from it.

Paths such as out_dir are relative to the current working directory.
"""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from gaia.engine.sim import HomeSpec
from gaia.home.archetypes import APARTMENTS, ARCHETYPES
from gaia.rng import derive_seed, rng_for
from gaia.world.cities import CITIES


class RunConfig(BaseModel):
    world_version: str
    seed: int
    homes: int = Field(gt=0)
    days: int = Field(gt=0)
    start_date: date
    cities: list[str] = Field(min_length=1)
    archetypes: list[str] = Field(default_factory=lambda: list(APARTMENTS), min_length=1)
    festivals: list[date] = Field(default_factory=list)
    split: str = "train"
    out_dir: Path
    workers: int = Field(default=0, ge=0)
    ets_dir: Path | None = None  # anonymised ETS layouts (scripts/import_ets.py output)

    @field_validator("cities")
    @classmethod
    def _known_cities(cls, value: list[str]) -> list[str]:
        unknown = [c for c in value if c not in CITIES]
        if unknown:
            raise ValueError(f"unknown cities: {unknown}")
        return value

    @field_validator("archetypes")
    @classmethod
    def _known_archetypes(cls, value: list[str]) -> list[str]:
        unknown = [a for a in value if a not in ARCHETYPES and a != "ets"]
        if unknown:
            raise ValueError(f"unknown archetypes: {unknown}")
        return value


    @model_validator(mode="after")
    def _ets_needs_dir(self) -> "RunConfig":
        if "ets" in self.archetypes and self.ets_dir is None:
            raise ValueError("archetype 'ets' needs ets_dir set in the run config")
        return self


def load_config(path: Path) -> RunConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return RunConfig(**yaml.safe_load(fh))


def home_spec(cfg: RunConfig, index: int) -> HomeSpec:
    """Spec for home `index`. Depends only on (seed, index), not on the total home count."""
    home_seed = derive_seed(cfg.seed, "home", index)
    rng = rng_for(home_seed, "assign")
    city = cfg.cities[int(rng.integers(len(cfg.cities)))]
    archetype = cfg.archetypes[int(rng.integers(len(cfg.archetypes)))]
    return HomeSpec(
        home_id=f"h{cfg.seed}-{index:06d}",
        home_seed=home_seed,
        city=city,
        archetype=archetype,
        start_date=cfg.start_date,
        days=cfg.days,
        festivals=frozenset(cfg.festivals),
        ets_dir=str(cfg.ets_dir) if cfg.ets_dir else None,
    )
