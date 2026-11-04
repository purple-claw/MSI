"""Import converted ETS projects into anonymised layouts.

usage: python scripts/import_ets.py <knxdata_dir> [out_dir]   (default out_dir: data/ets_layouts)

Reads every KNXdata.json under knxdata_dir and writes <project_id>.json per project plus
summary.json (counts only). Output stays local: data/ is git-ignored.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaia.ets.importer import import_knxdata, inventory_stats, is_residential  # noqa: E402


def main(argv: list[str]) -> int:
    src = Path(argv[1])
    out = Path(argv[2]) if len(argv) > 2 else Path("data/ets_layouts")
    out.mkdir(parents=True, exist_ok=True)
    layouts = [import_knxdata(p) for p in sorted(src.rglob("KNXdata.json"))]
    residential = [lay for lay in layouts if is_residential(lay)]
    for lay in layouts:
        (out / f"{lay.project_id}.json").write_text(lay.to_json(), encoding="utf-8")
    total = sum(lay.stats["loads_total"] for lay in layouts)
    unknown = sum(lay.stats["loads_unknown_type"] for lay in layouts)
    unmapped = sum(lay.stats["loads_unmapped_room"] for lay in layouts)
    summary = {
        "projects": len(layouts),
        "residential": len(residential),
        "loads_total": total,
        "unknown_type_share": round(unknown / total, 4) if total else 0.0,
        "unmapped_room_share": round(unmapped / total, 4) if total else 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(json.dumps(inventory_stats(residential), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
