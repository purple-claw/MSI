"""G1 throughput check: 4 homes x 365 days on one worker. Prints seconds per home-year."""

import tempfile
from datetime import date
from pathlib import Path

from gaia.cli import generate
from gaia.config import RunConfig

HOMES = 4
DAYS = 365


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = RunConfig(
            world_version="bench",
            seed=1,
            homes=HOMES,
            days=DAYS,
            start_date=date(2026, 1, 1),
            cities=["hyderabad"],
            out_dir=Path(tmp),
            workers=1,
        )
        summary = generate(cfg)
    seconds_per_home_year = summary["seconds"] / summary["home_years"]
    print(f"homes={HOMES} days={DAYS} workers=1 total_seconds={summary['seconds']} "
          f"events={summary['events']}")
    print(f"seconds_per_home_year={seconds_per_home_year:.2f}")


if __name__ == "__main__":
    main()
