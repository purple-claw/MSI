# GAIA G0+G1 Implementation Plan: schema and walking skeleton

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the MSI Event Schema v0 and the load-class registry (G0). Then build the smallest end-to-end GAIA:
one Indian 2BHK apartment, two personas, sun-driven lighting, wall/app/voice interactions, forgotten lights,
written as deterministic Parquet shards with throughput measured (G1).

**Architecture:** A pure-Python package `gaia/` with one module per spec layer (world, home, household, behaviour,
interaction, engine, recorder). Each home is simulated independently by `simulate_home` from a per-home seed, so
generation parallelises across processes. Random streams are derived per (home seed, layer name) so layers never disturb each other.

**Tech Stack:** Python ≥3.12 (this PC has 3.13.5), uv, numpy, pyarrow, pydantic v2, PyYAML, astral (sun position), pytest, hypothesis.

**Spec:** `docs/plans/gaia-spec.md` (§3.1–3.5, §3.7 latency only, §4, §7, §10, §11 G0–G1). Roadmap: `docs/research/msi-research-roadmap.md`.

## Global Constraints

- Everything runs on this PC (`/home/iris/Miantic/MSI`). Nothing runs on a board. Colab is only for later GPU work.
- Git author `Nitin.J <hawkr173@gmail.com>` (local config). **No AI or Claude attribution** in commits, code or docs.
- All timestamps are `ts_utc` int64 **milliseconds**. India local time = UTC + **330** minutes (no DST).
- Every synthetic row has `domain="synthetic"`.
- `protocol` is recorded but never used for behaviour decisions.
- Same config + same `world_version` ⇒ byte-identical Parquet output.
- Home *i*'s output depends only on (run seed, *i*), never on how many homes are in the run.
- Python package root: `gaia/`; tests in `tests/`; run everything with `uv run pytest`.
- Commit style: conventional commits, one concern per commit.

## Review Focus

1. **Sleep after midnight** (a persona's sleep time of 00:30): the day plan must wrap into the next local day, not produce a negative or overlapping activity. Test in Task 5.
2. **IST vs UTC day boundary**: local 00:00 IST is 18:30 UTC of the previous day. Day plans are in local time and events in UTC. An off-by-330-minutes bug shifts every routine. Test in Task 3.
3. **Home-count independence**: home 7 must be identical whether the run has 8 homes or 800. Test in Task 9.
4. **Bad config** (`homes<=0`, `days<=0`, unknown city): reject with a clear pydantic error before any work. Test in Task 9.
5. **Re-running into an existing output dir**: overwrite the home's shard cleanly. Never append duplicate rows. Test in Task 8.

---

### Task 1: Project scaffold, deterministic RNG, MSI Event Schema v0 (G0)

**Files:**
- Create: `pyproject.toml`, `gaia/__init__.py`, `gaia/rng.py`, `gaia/schema/__init__.py`, `gaia/schema/events.py`
- Test: `tests/test_rng.py`, `tests/schema/test_events.py`

**Interfaces:**
- Produces:
  - `gaia.rng.derive_seed(*parts: str | int) -> int`: stable 63-bit int from `hashlib.blake2b` over `"/".join(map(str, parts))`. Never Python `hash()`.
  - `gaia.rng.rng_for(home_seed: int, layer: str) -> numpy.random.Generator`, which is `np.random.default_rng(derive_seed(home_seed, layer))`.
  - Enums (`str, Enum`) in `gaia.schema.events`:
    `EventKind{command,state,scene_fired,schedule_fired,automation_edit,sensor,system,msi_proposal,msi_action,user_feedback}`,
    `Source{wall,app,voice,homekit,scene,schedule,msi,system,unknown}`,
    `Protocol{knx,matter,zigbee,wifi,other}`,
    `Domain{synthetic,real}`.
  - `@dataclass(frozen=True, slots=True) class Event` with fields exactly as spec §4.1:
    `ts_utc:int, tz_offset_min:int, home_id:str, event_id:str, kind:EventKind, room_id:str|None, room_type:str|None,
    load_id:str|None, load_class:str|None, capability:str|None, value:float|str|None, prev_value:float|str|None,
    source:Source, actor:str, protocol:Protocol|None, ref_id:str|None, domain:Domain`.
  - `EVENT_ARROW_SCHEMA: pyarrow.Schema` (enums stored as strings; `value`/`prev_value` stored as string, with floats formatted by `repr`).
  - `events_to_table(events: list[Event]) -> pyarrow.Table`.

- [ ] **Step 1: Scaffold.** `uv init --package --name gaia --python 3.12` (adjust so the package dir is `gaia/`), then `uv add numpy pyarrow pydantic pyyaml astral` and `uv add --dev pytest hypothesis`. Check: `uv run python -c "import gaia"` exits 0.
- [ ] **Step 2: Write failing tests**

```python
# tests/test_rng.py
from gaia.rng import derive_seed, rng_for
def test_derive_seed_stable_and_distinct():
    assert derive_seed(42, "world") == derive_seed(42, "world")
    assert derive_seed(42, "world") != derive_seed(42, "household")
    assert 0 <= derive_seed(1, "x") < 2**63
def test_rng_for_reproducible():
    assert rng_for(7, "behaviour").random() == rng_for(7, "behaviour").random()

# tests/schema/test_events.py
from gaia.schema.events import Event, EventKind, Source, Protocol, Domain, events_to_table, EVENT_ARROW_SCHEMA
def _ev(**kw):
    base = dict(ts_utc=1_790_000_000_000, tz_offset_min=330, home_id="h1", event_id="e1", kind=EventKind.command,
                room_id="r1", room_type="living", load_id="l1", load_class="light.onoff", capability="power",
                value=1.0, prev_value=0.0, source=Source.wall, actor="p1", protocol=Protocol.knx, ref_id=None,
                domain=Domain.synthetic)
    base.update(kw); return Event(**base)
def test_table_matches_schema():
    t = events_to_table([_ev(), _ev(event_id="e2", kind=EventKind.state, ref_id="e1")])
    assert t.schema.equals(EVENT_ARROW_SCHEMA)
    assert t.column("kind").to_pylist() == ["command", "state"]
    assert t.column("value").to_pylist() == ["1.0", "1.0"]
def test_event_is_frozen():
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        _ev().value = 0.0
```

- [ ] **Step 3: Run.** `uv run pytest tests/test_rng.py tests/schema -v` → FAIL (import errors).
- [ ] **Step 4: Implement** `gaia/rng.py` and `gaia/schema/events.py` per the Interfaces block.
- [ ] **Step 5: Run.** Same command → PASS.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "feat(gaia): scaffold, deterministic rng, MSI event schema v0"`

---

### Task 2: Load-class registry v0 (G0)

**Files:**
- Create: `gaia/registry/__init__.py`, `gaia/registry/load_classes.py`
- Test: `tests/registry/test_load_classes.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class LoadClass: name:str; capabilities:tuple[str,...]; rated_watts:float; is_sensor:bool=False`
  - `REGISTRY: dict[str, LoadClass]`, `get_load_class(name:str) -> LoadClass` (raises `KeyError` with the unknown name in the message)
  - v0 entries (name → capabilities, watts):
    `light.onoff`→(power) 12 · `light.dim`→(power,level) 12 · `light.cct`→(power,level,cct) 15 · `light.rgb`→(power,level,color) 15 ·
    `fan.speed`→(power,fan_speed) 75 · `ac`→(power,mode,setpoint,fan_speed) 1500 · `curtain.position`→(position) 60 ·
    `blind.tilt`→(position,tilt) 40 · `plug`→(power) 100 · `geyser`→(power) 2000 · `water_pump`→(power) 750 ·
    `exhaust`→(power) 40 · `tv_media`→(power) 120 · `mosquito_repellent`→(power) 5 · `scene_button`→(press) 0 ·
    sensors (is_sensor=True, 0 W): `motion`→(occupancy) · `door`→(contact) · `temp`→(temperature) · `humidity`→(humidity) · `lux`→(illuminance) · `energy_meter`→(power_w,energy_wh)

- [ ] **Step 1: Failing tests**: `test_all_spec_classes_present` (the set of names above equals `REGISTRY.keys()`), `test_light_dim_caps` (`get_load_class("light.dim").capabilities == ("power","level")`), `test_unknown_raises` (`pytest.raises(KeyError, match="toaster")`), `test_sensors_zero_watts`.
- [ ] **Step 2: Run** `uv run pytest tests/registry -v` → FAIL.
- [ ] **Step 3: Implement** `gaia/registry/load_classes.py`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): load-class registry v0`

---

### Task 3: World: city presets, sun and outdoor lux, day type

**Files:**
- Create: `gaia/world/__init__.py`, `gaia/world/cities.py`, `gaia/world/sun.py`, `gaia/world/calendar.py`
- Test: `tests/world/test_cities.py`, `tests/world/test_sun.py`, `tests/world/test_calendar.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class City: name:str; lat:float; lon:float; tz_offset_min:int = 330`
  - `CITIES: dict[str, City]` with `hyderabad(17.3850, 78.4867)`, `bengaluru(12.9716, 77.5946)`, `chennai(13.0827, 80.2707)`, `mumbai(19.0760, 72.8777)`, `delhi(28.6139, 77.2090)`
  - `sun_altitude_deg(city: City, ts_utc_ms: int) -> float` via `astral.sun.elevation`
  - `outdoor_lux(altitude_deg: float) -> float`: 0 if alt < −6; `400*(alt+6)/6` for −6 ≤ alt < 0; `400 + 110000*sin(radians(alt))` for alt ≥ 0 (clear sky; weather comes in G2)
  - `local_date(ts_utc_ms:int, tz_offset_min:int) -> datetime.date`; `local_midnight_utc_ms(d: date, tz_offset_min:int) -> int`
  - `DayType(str, Enum){workday, weekend, festival}`; `day_type(d: date, festivals: frozenset[date]) -> DayType` (festival wins, Sat/Sun = weekend)

- [ ] **Step 1: Failing tests**
  - `test_noon_bright_midnight_dark`: Hyderabad, 2026-06-21 at 06:30 UTC (12:00 IST): `outdoor_lux(alt) > 50_000`. At 18:30 UTC (00:00 IST): `== 0`.
  - `test_lux_monotonic_in_altitude`: hypothesis over alt in [−10, 90]: `outdoor_lux(a) <= outdoor_lux(a+1)`.
  - `test_ist_day_boundary` (Review Focus 2): `local_date(local_midnight_utc_ms(date(2026,11,4),330),330) == date(2026,11,4)`, and 1 ms earlier is `date(2026,11,3)`. Also `local_midnight_utc_ms(date(2026,11,4),330)` equals the UTC ms of `2026-11-03T18:30:00Z`.
  - `test_day_type`: 2026-11-07 (Sat) → weekend. 2026-11-04 (Wed) → workday. Wed in festivals → festival.
- [ ] **Step 2: Run** `uv run pytest tests/world -v` → FAIL.
- [ ] **Step 3: Implement** the three modules.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): world layer - cities, sun/lux, IST calendar`

---

### Task 4: Home: manifest and `apartment_2bhk` archetype

**Files:**
- Create: `gaia/home/__init__.py`, `gaia/home/manifest.py`, `gaia/home/archetypes.py`
- Test: `tests/home/test_archetypes.py`

**Interfaces:**
- Consumes: `get_load_class` (Task 2), `rng_for` (Task 1), `Protocol` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class Room: room_id:str; room_type:str; window_factor:float`
  - `@dataclass(frozen=True) class Load: load_id:str; room_id:str; load_class:str; protocol:Protocol; is_main_light:bool`
  - `@dataclass(frozen=True) class HomeManifest: home_id:str; archetype:str; city:str; rooms:tuple[Room,...]; adjacency:dict[str,tuple[str,...]]; loads:tuple[Load,...]`, plus `to_json() -> str` (sorted keys, stable), `loads_in(room_id) -> tuple[Load,...]`, `main_light(room_id) -> Load | None`
  - `ARCHETYPES: dict[str, Callable[[str, str, numpy.random.Generator], HomeManifest]]` (args: `home_id, city, rng`). G1 registers only `"apartment_2bhk"`.
  - `apartment_2bhk` layout: rooms `living`(window_factor 0.03), `kitchen`(0.02), `master_bedroom`(0.02), `bedroom_2`(0.02), `bathroom`(0.005). Every room is adjacent only to `living` (hub). Loads: every room has one main light (`light.dim` with p=0.5 else `light.onoff`). Living, master and bedroom_2 have `fan.speed`. Master has `ac` with p=0.8, bedroom_2 with p=0.4. Living has `tv_media`. Kitchen has `exhaust`. Bathroom has `geyser` with p=0.5. Protocol per load: knx 0.6, zigbee 0.25, matter 0.15. IDs: `room_id = f"{home_id}:{room_type}"`, `load_id = f"{room_id}:{load_class}:{n}"`.

- [ ] **Step 1: Failing tests**: `test_every_room_has_one_main_light` (200 seeds), `test_rooms_connected_via_living`, `test_same_seed_same_manifest_json`, `test_load_classes_registered` (every `load.load_class` is in `REGISTRY`), `test_protocol_mix_roughly_matches` (over 500 homes, knx share in [0.55, 0.65]).
- [ ] **Step 2: Run** `uv run pytest tests/home -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): home manifest and apartment_2bhk archetype`

---

### Task 5: Household: personas and day planner

**Files:**
- Create: `gaia/household/__init__.py`, `gaia/household/personas.py`, `gaia/behaviour/__init__.py`, `gaia/behaviour/day_plan.py`
- Test: `tests/household/test_personas.py`, `tests/behaviour/test_day_plan.py`

**Interfaces:**
- Consumes: `DayType` (Task 3), `rng_for` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class Persona: role:str; wake_min:tuple[float,float]; sleep_min:tuple[float,float]; leave_min:tuple[float,float]|None; return_min:tuple[float,float]|None; forgetfulness:float; tech_affinity:dict[str,float]; light_threshold_lux:float` (tuples are (mean, sd) in minutes after local midnight; `sleep_min` mean may exceed 1440 to mean after midnight)
  - `PERSONAS` with exactly two G1 entries (TUS calibration replaces these numbers in G2):
    - `working_professional`: wake (405, 25), sleep (1395, 30), leave (525, 20), return (1140, 40), forgetfulness 0.15, tech_affinity {wall 0.6, app 0.3, voice 0.1}, threshold 150
    - `homemaker`: wake (360, 20), sleep (1350, 25), leave None, return None, forgetfulness 0.08, tech_affinity {wall 0.8, app 0.1, voice 0.1}, threshold 120
  - `@dataclass(frozen=True) class Household: members:tuple[tuple[str, Persona], ...]` (person_id, persona). `make_household(home_id:str, rng) -> Household` returns either {professional, homemaker} (p=0.6) or {professional, professional} (p=0.4). Person ids are `f"{home_id}:p{n}"`.
  - `@dataclass(frozen=True) class Activity: name:str; room_type:str|None; start_min:int; end_min:int` (minutes from local midnight of the plan day; `room_type=None` means away; `end_min` may exceed 1440 only for the final `sleep`)
  - `plan_day(persona: Persona, day_type: DayType, rng) -> list[Activity]`: contiguous cover from `0` to `wake`, then to the next day's `sleep` end. Templates:
    - professional, workday: `sleep(master_bedroom)` 0→wake · `bath(bathroom)` 20 min · `breakfast(kitchen)` 20 min · `living` until leave · `away(None)` leave→return · `relax(living)` · `dinner(kitchen)` 30 min starting at return+60 · `tv(living)` until sleep · `sleep(master_bedroom)` sleep→1440+wake_next(=wake)
    - professional, weekend or festival: same without away. Wake +60 min. `relax(living)` all day with `lunch(kitchen)` 30 min at 780.
    - homemaker, any day: `sleep` · `bath` 20 · `cook(kitchen)` 60 · `living` · `cook(kitchen)` 60 at 720 · `rest(master_bedroom)` 60 at 840 · `living` · `cook(kitchen)` 60 at 1170 · `tv(living)` · `sleep`
    - Sampled times are rounded to whole minutes and clamped so the order always holds (each activity ≥ 5 min).

- [ ] **Step 1: Failing tests**
  - `test_plan_contiguous_and_ordered`: hypothesis over seeds × personas × day types: `acts[0].start_min == 0`, each `acts[i].end_min == acts[i+1].start_min`, every duration ≥ 5.
  - `test_sleep_after_midnight_wraps` (Review Focus 1): a persona with `sleep_min=(1470, 0)` (00:30) yields a final `sleep` activity starting at 1470 and no activity with `end_min < start_min`.
  - `test_professional_workday_has_away`, `test_weekend_has_no_away`.
  - `test_household_ids_stable`: same seed gives the same person ids and roles.
- [ ] **Step 2: Run** `uv run pytest tests/household tests/behaviour -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): personas, household, day planner`

---

### Task 6: Needs and interaction

**Files:**
- Create: `gaia/behaviour/needs.py`, `gaia/interaction/__init__.py`, `gaia/interaction/choose.py`
- Test: `tests/behaviour/test_needs.py`, `tests/interaction/test_choose.py`

**Interfaces:**
- Consumes: `Persona` (Task 5), `Room` (Task 4), `Source` (Task 1).
- Produces:
  - `indoor_lux(outdoor_lux: float, room: Room) -> float`, which is `outdoor_lux * room.window_factor`
  - `light_needed(indoor: float, persona: Persona, activity_name: str) -> bool`: False when `activity_name == "sleep"`, else `indoor < persona.light_threshold_lux`
  - `choose_source(persona: Persona, rng) -> Source` (samples from `tech_affinity` keys mapped to `Source.wall/app/voice`)
  - `forgets(persona: Persona, rng) -> bool` (Bernoulli `forgetfulness`)

- [ ] **Step 1: Failing tests**: `test_no_light_when_sleeping`; `test_dark_room_needs_light` (indoor 10, threshold 150 → True); `test_choose_source_distribution` (10k draws for the professional: wall share in [0.57, 0.63]); `test_forgets_rate` (10k draws, homemaker rate in [0.06, 0.10]).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): light need and interaction choice`

---

### Task 7: Engine: `simulate_home`

**Files:**
- Create: `gaia/engine/__init__.py`, `gaia/engine/sim.py`, `gaia/engine/protocols.py`
- Test: `tests/engine/test_sim.py`, `tests/engine/test_invariants.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `protocols.latency_ms(protocol: Protocol, rng) -> int`: uniform knx [30,80], zigbee [50,150], matter [40,120], wifi [80,300], other [50,200]
  - `@dataclass(frozen=True) class HomeSpec: home_id:str; home_seed:int; city:str; archetype:str; start_date:date; days:int; festivals:frozenset[date]`
  - `@dataclass(frozen=True) class GtCause: event_id:str; cause:str; person_id:str|None; activity:str|None` (cause ∈ `need_light`, `leave_room`, `sleep`, `forgot_cleanup`)
  - `@dataclass(frozen=True) class GtWaste: load_id:str; person_id:str; start_ts_utc:int; end_ts_utc:int|None` (a forgotten light interval)
  - `@dataclass(frozen=True) class ContextRow: ts_utc:int; home_id:str; outdoor_lux:float; sun_altitude_deg:float`
  - `@dataclass class SimResult: manifest:HomeManifest; household:Household; events:list[Event]; context:list[ContextRow]; gt_cause:list[GtCause]; gt_waste:list[GtWaste]`
  - `simulate_home(spec: HomeSpec) -> SimResult`

**Algorithm** (the tests don't pin this down, so it's fixed here):
1. Build the manifest (`ARCHETYPES[spec.archetype]`, rng layer `"home"`) and the household (layer `"household"`).
2. For each local day, plan each member's day (layer `f"plan/{person_id}/{day_index}"`). Combine the plans into per-person intervals in UTC ms (`local_midnight_utc_ms` + minutes). A plan's trailing sleep overlapping the next day's leading sleep is merged.
3. Step a 1-minute clock over the whole range. Each tick: compute the sun and lux once (append a `ContextRow` every 15 minutes, not every minute). Each person's current room comes from their interval.
4. **Room entry** or **every tick while present**: if the room's main light is off and `light_needed` holds, emit a `command` (source from `choose_source`, actor = person, ts = tick + rng jitter 0–59 s), then a `state` event at `+latency_ms` with `ref_id` = the command id. Record `GtCause(need_light)`.
5. **Room exit** (the person goes to another room or away, and no other member remains in the room): if the light is on, `forgets` → leave it on and open a `GtWaste` interval. Otherwise emit off command + state with `GtCause(leave_room)`.
6. **Sleep start**: turn off the bedroom light (`GtCause(sleep)`). Also, if this person is the last member awake, each light still on in an empty room is turned off with p = 0.5 (`source=app`, `GtCause(forgot_cleanup)`), closing its `GtWaste`.
7. Event ids are `f"{home_id}:{counter:09d}"` in emission order. Sort events by `(ts_utc, event_id)` at the end.

- [ ] **Step 1: Failing tests** (`tests/engine/test_sim.py`)

```python
def _spec(seed=1, days=7):
    return HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026,11,2), days=days, festivals=frozenset())
def test_produces_events_and_context():
    r = simulate_home(_spec())
    assert len(r.events) > 50 and len(r.context) == 7*24*4
def test_deterministic():
    a, b = simulate_home(_spec(5)), simulate_home(_spec(5))
    assert a.events == b.events and a.gt_cause == b.gt_cause
def test_lights_on_after_sunset_not_noon():
    r = simulate_home(_spec(days=14))
    on_hours_ist = [((e.ts_utc//60000 + 330) // 60) % 24 for e in r.events
                    if e.kind == EventKind.command and e.value == 1.0 and e.room_type == "living"]
    assert sum(17 <= h <= 23 for h in on_hours_ist) > sum(11 <= h <= 14 for h in on_hours_ist)
```

  `tests/engine/test_invariants.py` (hypothesis, `max_examples=25`, seeds 0–10⁶, days=3):
  - `test_ts_monotonic`: event timestamps never decrease.
  - `test_every_state_has_command`: each `state` event's `ref_id` is an earlier `command` id in the same home.
  - `test_no_redundant_state`: per load, consecutive `state` values alternate (never on→on or off→off).
  - `test_gt_cause_covers_commands`: every `command` event_id has exactly one `GtCause`.
  - `test_waste_intervals_valid`: `end_ts_utc is None or end_ts_utc > start_ts_utc`.
- [ ] **Step 2: Run** `uv run pytest tests/engine -v` → FAIL.
- [ ] **Step 3: Implement** `protocols.py` and `sim.py` per the algorithm.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): per-home discrete simulation engine`

---

### Task 8: Recorder: Parquet shards and manifest

**Files:**
- Create: `gaia/recorder/__init__.py`, `gaia/recorder/parquet.py`, `gaia/schema/context.py`
- Test: `tests/recorder/test_parquet.py`

**Interfaces:**
- Consumes: `SimResult`, `events_to_table`.
- Produces:
  - `CONTEXT_ARROW_SCHEMA`, `GT_CAUSE_ARROW_SCHEMA`, `GT_WASTE_ARROW_SCHEMA` (in `gaia/schema/context.py`)
  - `shard_dir(out_dir: Path, world_version: str, split: str, home_id: str) -> Path`, which is `out_dir/f"world={world_version}"/f"split={split}"/f"home={home_id}"`
  - `write_home(result: SimResult, out_dir: Path, world_version: str, split: str) -> Path` writes `events.parquet`, `context.parquet`, `gt_cause.parquet`, `gt_waste.parquet` and `manifest.json` (manifest JSON plus household roles plus `world_version`). The shard is written to a temporary sibling dir and then `os.replace`d over the old one (Review Focus 5). Parquet is written with `compression="zstd"` and no file-level timestamps, so bytes are stable.

- [ ] **Step 1: Failing tests**: `test_roundtrip_event_count` (written rows == `len(result.events)`); `test_rewrite_overwrites_not_appends` (Review Focus 5: write twice, row count unchanged, no leftover temp dir); `test_bytes_stable` (two writes of the same result give the same `events.parquet` sha256).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(gaia): parquet recorder with atomic shard writes`

---

### Task 9: Run config, CLI, parallel generation, throughput

**Files:**
- Create: `gaia/config.py`, `gaia/cli.py`, `gaia/__main__.py`, `configs/g1-smoke.yaml`, `scripts/bench_g1.py`
- Test: `tests/test_config.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `simulate_home`, `write_home`, `CITIES`, `ARCHETYPES`, `derive_seed`.
- Produces:
  - pydantic `RunConfig`: `world_version:str`, `seed:int`, `homes:int (gt 0)`, `days:int (gt 0)`, `start_date:date`, `cities:list[str]` (each must be in `CITIES`), `archetypes:list[str]=["apartment_2bhk"]` (each in `ARCHETYPES`), `festivals:list[date]=[]`, `split:str="train"`, `out_dir:Path`, `workers:int=0` (0 means `os.cpu_count()`)
  - `load_config(path: Path) -> RunConfig` (YAML)
  - `home_spec(cfg: RunConfig, index: int) -> HomeSpec`: `home_id=f"h{cfg.seed}-{index:06d}"`, `home_seed=derive_seed(cfg.seed, "home", index)`. City and archetype are chosen with `rng_for(home_seed, "assign")`. This depends only on (seed, index) (Review Focus 3).
  - `generate(cfg: RunConfig) -> dict` runs a `ProcessPoolExecutor` over indices and returns `{"homes": n, "events": total, "seconds": t, "home_years_per_hour": x}`
  - CLI: `uv run python -m gaia generate <config.yaml>` prints that dict as JSON.
  - `configs/g1-smoke.yaml`: world_version `g1`, seed 2026, homes 8, days 30, start_date 2026-11-02, cities [hyderabad, bengaluru], out_dir `data/g1-smoke`.

- [ ] **Step 1: Failing tests**
  - `test_rejects_bad_config` (Review Focus 4): `homes=0`, `days=-1` and `cities=["atlantis"]` each raise `pydantic.ValidationError`.
  - `test_home_spec_independent_of_count` (Review Focus 3): `home_spec(cfg_with_8_homes, 7) == home_spec(cfg_with_800_homes, 7)`.
  - `test_cli_smoke(tmp_path)`: a run with homes=2, days=2 into `tmp_path` exits 0 and writes 2 shard dirs, each containing `events.parquet`.
  - `test_run_twice_identical(tmp_path)`: two runs → identical sha256 of every `events.parquet`.
- [ ] **Step 2: Run** `uv run pytest tests/test_config.py tests/test_cli.py -v` → FAIL.
- [ ] **Step 3: Implement** config, CLI and `generate`.
- [ ] **Step 4: Run** → PASS. Then the full suite `uv run pytest -q` → all PASS.
- [ ] **Step 5: Measure throughput.** `scripts/bench_g1.py` simulates 4 homes × 365 days on 1 worker and prints seconds per home-year.
  Run `uv run python scripts/bench_g1.py`. Record the number in `docs/plans/gaia-spec.md` §7 as "G1 measured: X s per home-year (target < 10 s)". If it misses the target, note it. Don't optimise in this plan.
- [ ] **Step 6: Smoke run.** `uv run python -m gaia generate configs/g1-smoke.yaml` → JSON with `homes: 8` and `events > 0`.
- [ ] **Step 7: Commit** `feat(gaia): run config, CLI, parallel generation, G1 benchmark`

---

## Done means (G1 exit gate from the spec)
- `uv run pytest -q` all green, including the hypothesis invariants.
- `configs/g1-smoke.yaml` produces 8 deterministic home shards.
- The throughput number is recorded in the spec.
- Not in this plan (G2+): weather, other archetypes, TUS calibration, scenes and schedules, drift, faults beyond latency, thermal and energy, fidelity report, closed loop.
