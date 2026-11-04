"""G2a speed check: one home-year per archetype (1RK, 3BHK, villa) on one worker.

Runs the same path as `gaia generate` (simulate + parquet write) and prints seconds per
home-year for each archetype, the mean over the three, and the event count.
"""

import tempfile
from datetime import date
from pathlib import Path

from gaia.cli import generate
from gaia.config import RunConfig

ARCHETYPES = ("apartment_1rk", "apartment_3bhk", "villa")
DAYS = 365


def bench_one(archetype: str) -> tuple[float, int]:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = RunConfig(
            world_version="bench",
            seed=1,
            homes=1,
            days=DAYS,
            start_date=date(2026, 1, 1),
            cities=["hyderabad"],
            archetypes=[archetype],
            out_dir=Path(tmp),
            workers=1,
        )
        summary = generate(cfg)
    return summary["seconds"] / summary["home_years"], summary["events"]


def main() -> None:
    rates = []
    for archetype in ARCHETYPES:
        seconds_per_home_year, events = bench_one(archetype)
        rates.append(seconds_per_home_year)
        print(f"archetype={archetype} days={DAYS} workers=1 events={events} "
              f"seconds_per_home_year={seconds_per_home_year:.2f}")
    print(f"mean_seconds_per_home_year={sum(rates) / len(rates):.2f}")


if __name__ == "__main__":
    main()
