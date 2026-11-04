# GAIA G2a Implementation Plan: rich loads, behaviour policies, labels v1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn GAIA from "one main light per room" into realistic premium Indian homes:
- about 60 protocol-neutral load classes (lighting systems and fixtures, AV, climate, shading, utility, security, sensors, keypads)
- several loads per room
- a behaviour policy per load domain
- hierarchical labels with provenance and confidence on every event
- homes built from anonymised real ETS layouts
- a dataset quality gate

**Architecture:**
- **Data-driven registry.** A YAML file describes control classes (capabilities, ranges, Matter device type, power with source, waste rule) and fixture tags.
- **State model.** A `DeviceState` validates and applies capability changes.
- **Policies.** The engine stops hard-coding lights. Each tick it builds a read-only `TickContext`, asks an ordered list of `Policy` objects for `Intent`s, and an `Actuator` applies them.
- **Actuator.** It dedupes, validates, serialises timestamps, emits the command and state events, and writes one label row per command.
- **Waste.** It is generalised per class waste rule.

**Tech:** unchanged (Python ≥3.12, uv, numpy, pyarrow, pydantic, PyYAML, astral, pytest, hypothesis).

**Spec:** `docs/plans/gaia-spec.md` §3.2–3.7, §4, §8, §10. Decisions D15, D30, D34, D43 and P33 in agres. Research inputs (scratch, not committed): `.superpowers/research/g2-load-taxonomy.md`, `.superpowers/research/g2-labelling.md`.

## Global Constraints
- Runs only on this PC. Git author Nitin.J. **No AI attribution anywhere.** Conventional commits.
- Timestamps are int64 UTC ms, IST +330. Every row has a `domain`.
- Determinism: same config + world version ⇒ byte-identical shards. Home *i* depends only on (seed, *i*).
- Every random draw comes from a named layer stream via `rng_for(home_seed, name)`. A new policy gets its own stream `policy/<name>`, so adding a policy never shifts another policy's draws.
- **No unsourced numbers presented as facts.** Every power or usage number in YAML carries `source:` (a URL) or `source: assumption`.
- **Client ETS data never enters git.** No `.knxproj`, client names or raw KNXdata in the repo (P33). Tests use hand-made fixtures with invented names.
- All G1 invariants keep holding for every load class: command before state, no redundant state, per-load ordering, waste-completeness, determinism.
- `.superpowers/` must be git-ignored (Task 1 adds it).

## Review Focus
1. **Semantics trap:** curtain position meaning is inverted across vendors. GAIA fixes **position 1.0 = fully open, 0.0 = closed** everywhere. Tilt is 0–1. Test in Task 2.
2. **Redundant commands:** a policy asking for a value the device already has must produce no event. Two policies targeting the same load and capability in one tick: the **first in policy order wins**. Test in Task 3.
3. **Range clamping:** a CCT outside the class range (e.g. 9000 K on a 2700–6500 K light) is clamped and labelled. It must not crash and must not emit an out-of-range value. Test in Task 2.
4. **Label coverage:** every command event has exactly one label row, and every label row references an existing event. Test in Task 3 and in the Task 12 checker.
5. **Privacy:** the ETS importer must never write a client project or room name into its output. Test in Task 11, with a fixture whose names contain a sentinel string.

---

### Task 1: Load registry v1 (data-driven) and protocol enum

**Files:**
- Create `gaia/registry/classes.yaml`, `gaia/registry/registry.py`
- Delete `gaia/registry/load_classes.py`
- Modify `gaia/schema/events.py` (Protocol), `.gitignore`, `gaia/home/archetypes.py` (new class names)
- Test `tests/registry/test_registry.py`

**Interfaces:**
- `Protocol` gains `lutron`, `dali`, `ip`, `ir` (keep `knx, matter, zigbee, wifi, other`).
- `@dataclass(frozen=True) class Capability: name:str; kind:str; lo:float|None; hi:float|None; unit:str|None; values:tuple[str,...]|None`
  - `kind` ∈ `binary | level | kelvin | hue | saturation | position | tilt | celsius | enum | count | event`.
  - `level`, `position`, `tilt`, `hue` and `saturation` are normalised 0–1. `kelvin` and `celsius` carry real ranges.
- `@dataclass(frozen=True) class LoadClass: name:str; domain:str; matter_device_type:str|None; capabilities:tuple[Capability,...]; power_w:dict[str,float]; power_source:str; waste_rule:str; is_sensor:bool; rooms:tuple[str,...]`
  - `power_w` maps a state key (`on`, `standby`, `active`) to watts.
  - `waste_rule` ∈ `unattended | overrun:<minutes> | none`.
- `REGISTRY: dict[str, LoadClass]` (loaded once from YAML), `get_load_class(name)` (KeyError naming the class), `FIXTURES: frozenset[str]`.
- **YAML content:** every class in research §3.1–3.7. Names:
  - **Lights:** `light.onoff`, `light.dimmable`, `light.tunable`, `light.rgbw`, `light.rgbcct`, `light.led_strip`, `light.festival`.
  - **Media:** `media.tv`, `media.stb_dth`, `media.streamer`, `media.speaker`, `media.soundbar`, `media.av_receiver`, `media.projector`, `media.screen_motorised`.
  - **Climate:** `climate.split_ac`, `climate.vrf_indoor`, `climate.cassette_ac`, `fan.bldc_ceiling`, `fan.ceiling`, `fan.exhaust`, `air.purifier`, `air.dehumidifier`.
  - **Shading:** `shade.curtain`, `shade.roller`, `shade.venetian`, `shade.skylight`.
  - **Water and utility:** `water.geyser`, `water.pump`, `water.ro_purifier`, `kitchen.chimney`, `kitchen.dishwasher`, `laundry.washing_machine`, `power.inverter_ups`, `ev.charger`, `plug.smart`, `misc.mosquito_repellent`.
  - **Security:** `lock.smart`, `doorbell.video`, `gate.motor`, `cctv.camera`.
  - **Sensors:** `sensor.motion`, `sensor.presence`, `sensor.contact`, `sensor.temperature`, `sensor.humidity`, `sensor.lux`, `sensor.co2`, `sensor.water_leak`, `meter.energy`.
  - **Keypads:** `keypad.scene`.
- **Media capabilities:**
  - `power` (binary)
  - `source` (enum: `stb`, `streamer`, `airplay`, `spotify`, `hdmi1`, `hdmi2`, `bluetooth`, `radio`, `tv_app`)
  - `playback` (enum: `playing`, `paused`, `stopped`)
  - `volume` (level)
  - `mute` (binary)
- **AC capabilities:** `power`, `mode` (enum: `cool`, `dry`, `fan`, `auto`), `setpoint` (celsius 16–30, `source: assumption`), `fan_speed` (count 1–5).
- **Lighting capabilities:**
  - `light.tunable` has `cct` kelvin 2700–6500 (`source: assumption`; retail Hue is about 2200–6500, research §3.1).
  - RGB classes have `hue`, `saturation`, `level` and `power`.
- **Power numbers:** use only the ones the research sourced, with their URLs:
  - Sonos idle 4.3 W
  - DTH set-top box 18 W on, 17 W standby (Tier C)
  - BLDC fan 26–35 W (use 30)
  - 1.5 T split AC 1315 W (use as `on`)
  - geyser 2000 W

  Everything else carries a plausible number marked `source: assumption`.
- **Fixtures:** `downlight`, `profile`, `cove`, `pendant`, `chandelier`, `track`, `wall_washer`, `step`, `landscape`, `facade`, `accent`, `strip`, `bulb`.

- [ ] Write failing tests:
  - `test_every_class_has_power_source` (`power_source` non-empty)
  - `test_ranges_valid` (`lo < hi` where both are set; enums have values)
  - `test_curtain_position_is_normalised` (kind `position`, 0–1)
  - `test_unknown_class_raises`
  - `test_sourced_numbers` (`media.speaker` standby 4.3; `water.geyser` on 2000; the source is a URL for both)
  - `test_protocol_enum_extended`
- [ ] Run `uv run pytest tests/registry -v` → FAIL.
- [ ] Implement:
  - Load YAML with `yaml.safe_load` into frozen dataclasses, validated by pydantic models. Use `importlib.resources` for the file path.
  - Update `archetypes.py` to the new names: `light.dim` → `light.dimmable`, `light.onoff` stays, `fan.speed` → `fan.ceiling`, `ac` → `climate.split_ac`, `tv_media` → `media.tv`, `exhaust` → `fan.exhaust`, `geyser` → `water.geyser`.
  - Add `.superpowers/` to `.gitignore`.
- [ ] Run the full suite → PASS. Fix any test that referenced the old names.
- [ ] Commit `feat(gaia): data-driven load registry v1 and protocol enum`.

---

### Task 2: Device state model

**Files:** create `gaia/devices/__init__.py`, `gaia/devices/state.py`; test `tests/devices/test_state.py`.

**Interfaces:**
- `class DeviceState`, built as `DeviceState(load_class: LoadClass)`.
  - Initial values: binary 0, level 0, kelvin at mid-range, position 0 (closed), enum = first value, celsius 24.
- `get(cap) -> float|str`
- `set(cap, value) -> tuple[float|str, float|str, bool] | None` returns `(new, prev, clamped)`. It returns `None` when `new == prev` after normalisation.
- `snapshot() -> dict`
- **Rules:**
  - Unknown capability → `ValueError`.
  - Numeric values are clamped into `[lo, hi]` with `clamped=True`.
  - Enum value not in `values` → `ValueError`.
  - Setting any non-`power` capability of a powered class while it's off implies power on. The caller sees one `set` per capability; the actuator turns the device on first. `DeviceState.requires_power(cap) -> bool` tells it.
- `class HomeDevices`: `{load_id: DeviceState}` built from a manifest. `get(load_id)` and `is_on(load_id)`. "On" means `power == 1`, or position > 0 for shades.

- [ ] Write failing tests:
  - `test_clamp_cct` (9000 → 6500, clamped)
  - `test_redundant_set_returns_none`
  - `test_enum_validation`
  - `test_position_semantics` (1.0 open)
  - `test_requires_power`
  - hypothesis `test_set_get_roundtrip_within_range`
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Commit `feat(gaia): device state model with ranges and clamping`.

---

### Task 3: Actuator and labels v1

**Files:**
- Create `gaia/labels/__init__.py`, `gaia/labels/schema.yaml`, `gaia/labels/model.py`, `gaia/engine/actuator.py`
- Modify `gaia/schema/context.py` (LABEL schema), `gaia/recorder/parquet.py`
- Test `tests/labels/test_labels.py`, `tests/engine/test_actuator.py`

**Interfaces:**
- `labels/schema.yaml` (versioned, `schema_version: 1`) defines the enums:
  - **activity:** every day-plan activity name plus `none`.
  - **intent:**
    - `illumination`
    - `thermal_comfort`
    - `air_quality`
    - `entertainment`
    - `privacy`
    - `hygiene`
    - `energy_saving`
    - `ambience`
    - `security`
    - `utility`
    - `routine`
    - `scene`
    - `schedule`
    - `noticed_waste`
    - `cleanup`
    - `guest`
    - `help_staff`
    - `ai_action`
    - `protocol_noise`
    - `system`
    - `unknown`
  - **trigger:**
    - `room_entry`
    - `room_exit`
    - `darkness`
    - `daylight`
    - `wake`
    - `sleep`
    - `activity_start`
    - `activity_end`
    - `presence_tick`
    - `notice`
    - `last_to_sleep`
    - `timer`
    - `scene`
    - `schedule`
  - **device_action:**
    - `turn_on`
    - `turn_off`
    - `set_level`
    - `set_cct`
    - `set_color`
    - `open`
    - `close`
    - `set_position`
    - `set_tilt`
    - `set_setpoint`
    - `set_mode`
    - `set_fan_speed`
    - `play`
    - `pause`
    - `stop`
    - `set_volume`
    - `mute`
    - `select_source`
    - `lock`
    - `unlock`
    - `scene_recall`
  - **provenance:** `sim_truth`, `sim_rule`, `heuristic_lf`, `model_inferred`, `human`, `imported_public`.
- `@dataclass(frozen=True) class Label` fields:
  - `event_id`, `person_id: str|None`, `room_id: str|None`, `activity`, `intent`, `trigger`, `device_action`
  - `provenance` (default `sim_truth`), `confidence: float` (1.0 for sim_truth), `source_ref: str` (the policy name), `schema_version: int`, `clamped: bool`
- `LABEL_ARROW_SCHEMA`. `validate_label(label) -> None` raises on any value outside the schema enums.
- `@dataclass(frozen=True) class Intent` fields:
  - `load_id`, `capability`, `value`
  - `person_id`, `activity`, `intent`, `trigger`, `device_action`
  - `source: Source|None` (None ⇒ the actuator draws `choose_source` for the person, or uses `Source.system` when there's no person)
- `class Actuator`, built as `Actuator(home_id, manifest, devices: HomeDevices, tz_offset_min, rng_engine, rng_interaction, rng_protocol, personas_by_id)`.
  - `apply(intents: list[Intent], tick_start_ms: int) -> None`
    - Dedupe per (load_id, capability): the first wins.
    - When `requires_power` and the device is off, first apply an implicit power-on intent with the same labels and `device_action=turn_on`.
    - `DeviceState.set`. If it returns None, skip.
    - Command ts = tick + jitter (engine rng), serialised per load as in G1. State ts = max(cmd + latency, prev state + 1).
    - Emit command and state events, plus one `Label` per command.
  - `events`, `labels` properties.
- `GtCause` is **removed**. The `labels` table replaces it in `SimResult`, and the recorder writes `labels.parquet` instead of `gt_cause.parquet`.

- [ ] Write failing tests:
  - `test_label_validation_rejects_unknown_intent`
  - `test_dedupe_first_wins`
  - `test_redundant_intent_no_event`
  - `test_implicit_power_on` (`set_cct` on an off tunable gives 2 commands and 2 labels)
  - `test_every_command_has_one_label`
  - `test_clamped_flag_propagates`
- [ ] Run → FAIL. Implement (engine wiring comes in Task 4; test the actuator standalone with a hand-built manifest). Run → PASS.
- [ ] Commit `feat(gaia): actuator and hierarchical labels v1`.

---

### Task 4: Policy framework; port G1 lighting into a policy

**Files:**
- Create `gaia/engine/context.py`, `gaia/policies/__init__.py`, `gaia/policies/base.py`, `gaia/policies/lighting_basic.py`
- Modify `gaia/engine/sim.py`
- Tests: adapt `tests/engine/*` to labels; create `tests/policies/test_lighting_basic.py`

**Interfaces:**
- `@dataclass(frozen=True) class PersonNow: person_id; persona; room_id|None; asleep: bool; activity: str; prev_room_id|None; prev_asleep: bool`
- `@dataclass(frozen=True) class TickContext`:
  - Time: `minute`, `tick_ms`, `local_minute_of_day`, `local_date`, `day_type`, `month`
  - Sun: `outdoor_lux`, `sun_altitude_deg`
  - People: `people: tuple[PersonNow,...]`
  - Home: `manifest`, `devices` (read-only view: `is_on`, `get`)
  - Helpers:
    - `awake_in(room_id) -> tuple[PersonNow,...]`
    - `entered() -> list[(PersonNow, room_id)]`
    - `left() -> list[(PersonNow, room_id)]`
    - `fell_asleep() -> list[PersonNow]`
    - `woke() -> list[PersonNow]`
- `class Policy(Protocol)`: `name: str`, `def tick(self, ctx: TickContext, rng) -> list[Intent]`, and an optional `def finish(self, ctx) -> None`.
- `POLICY_ORDER: list[type[Policy]]` in `gaia/policies/__init__.py`. Each policy gets `rng_for(home_seed, f"policy/{name}")`.
- **Waste tracking** moves into `gaia/engine/waste.py` (`WasteTracker`). It is driven generically by the registry `waste_rule`:
  - `unattended`: a powered load that is on while no awake member is in its room.
  - `overrun:N`: on for more than N minutes.
  - `GtWaste` gains `load_class`, and its `person_id` becomes nullable (the person who last left, if any).
- `lighting_basic` reproduces the G1 behaviour exactly, for **main lights only**:
  - entry or presence while dark → on
  - last awake leaver → off, or forgets
  - sleep → off
  - last-to-sleep cleanup with p=0.5

  Labels:
  - on: intent `illumination`, trigger `darkness` / `room_entry`
  - off on exit: intent `energy_saving`, trigger `room_exit`
  - sleep: intent `illumination`, trigger `sleep`, device_action `turn_off`
  - cleanup: intent `cleanup`, trigger `last_to_sleep`

- [ ] Write failing tests:
  - Port every G1 engine test and invariant to the policy engine, asserting the same behaviour. The waste-completeness, determinism, ordering and presence tests must keep passing.
  - Add `test_policy_stream_isolation`: adding a no-op policy that draws from its own stream leaves events byte-identical.
- [ ] Run → FAIL (imports). Implement. Run the full suite → PASS.
- [ ] Commit `refactor(gaia): policy engine with actuator; G1 lighting as a policy`.

---

### Task 5: Room inventory v1, tiers, villas, lighting systems

**Files:**
- Create `gaia/home/inventory.yaml`, `gaia/home/inventory.py`
- Modify `gaia/home/archetypes.py`, `gaia/home/manifest.py`
- Test `tests/home/test_inventory.py`

**Interfaces:**
- `Load` gains `fixture: str|None`, `circuit: str` (`main`, `accent`, `cove`, `task`, `night`, `decor`, or `none` for non-lights) and `system: str` (`knx_dali`, `lutron_ra3`, `lutron_caseta`, `zigbee_hue`, `matter_wifi`, `ir_gateway`, `ip`). `is_main_light` stays (it means `circuit == "main"`).
- `HomeManifest` gains `tier: str` (`basic`, `standard`, `premium`, `luxury`) and `lighting_system: str`.
- New archetypes:
  - `independent_house`: rooms living, dining, kitchen, pooja, bedroom_1..3, bathroom_1..3, balcony, passage, staircase; household 3–6.
  - `villa`: living, dining, kitchen, pooja, home_theatre, study, bedroom_1..4, bathroom_1..5, dressing, terrace, garden, lobby, staircase, utility; household 3–7.
  - Room types extend accordingly. `room_type` is the name without its index.
- Tier per archetype:
  - 1rk/1bhk → basic
  - 2bhk → standard
  - 3bhk → standard (p 0.6) or premium (p 0.4)
  - 4bhk/independent_house → premium
  - villa → luxury
- Lighting system per home by tier:
  - basic/standard: zigbee_hue 0.3, matter_wifi 0.4, knx_dali 0.3
  - premium: knx_dali 0.6, lutron_ra3 0.3, zigbee_hue 0.1
  - luxury: knx_dali 0.6, lutron_ra3 0.4

  The protocol per load follows the system (knx_dali → knx or dali for DT8 tunable; lutron_* → lutron; zigbee_hue → zigbee; matter_wifi → matter). AV loads use `ip`, IR-controlled ACs and set-top boxes use `ir`.
- `inventory.yaml` maps `tier → room_type →` a list of `{class, fixture, circuit, count: [min,max], p}`. All values are `source: assumption` until Task 11 calibrates them from ETS. Minimums:
  - Every lit room has exactly one `main` circuit.
  - Premium/luxury living and bedrooms add `cove` (`light.led_strip` or `light.dimmable`), `accent` (profile/track), and tunable downlights (`light.tunable`) with p ≥ 0.5.
  - `night` step lights in passages and staircases.
  - Living gets `media.tv` + `media.stb_dth` (p 0.8) + `media.streamer` (p 0.6) + `media.speaker` or `media.soundbar` (p by tier).
  - home_theatre gets `media.projector` + `media.screen_motorised` + `media.av_receiver`.
  - Bedrooms get `climate.split_ac` (p by tier) + `fan.bldc_ceiling` or `fan.ceiling` + `shade.curtain` (premium+).
  - Kitchen gets `kitchen.chimney`, `water.ro_purifier`, a `media.speaker` (p 0.3) and `fan.exhaust`.
  - Bathrooms get `fan.exhaust`, `water.geyser` (p 0.7).
  - Utility gets `water.pump`, `laundry.washing_machine`, `power.inverter_ups`.
  - Lobby/garden gets `lock.smart` (premium+), `doorbell.video`, `light.onoff` landscape, and `gate.motor` (villa).
  - Pooja room: `light.onoff` with fixture `accent` (lamp).
- `build_home(archetype, home_id, city, rng) -> HomeManifest` replaces `build_apartment` for all archetypes. `ARCHETYPES` keeps the same call signature. `APARTMENTS` is generalised to `ARCHETYPE_SPECS`.

- [ ] Write failing tests:
  - `test_every_lit_room_one_main_circuit` (all 7 archetypes × 100 seeds)
  - `test_luxury_living_has_layers` (over 200 villas, ≥ 50% have a cove or accent and ≥ 50% have tunable lights)
  - `test_loads_per_room_range` (villa living room has 4–25 loads)
  - `test_protocol_matches_system`
  - `test_all_classes_registered`
  - `test_manifest_json_stable`
- [ ] Run → FAIL. Implement. Run the full suite → PASS (lighting_basic keeps driving main lights only).
- [ ] Commit `feat(gaia): room inventory v1 with tiers, villas and lighting systems`.

---

### Task 6: Lighting policy v2 (layers, tunable, RGB, daylight, night lights)

**Files:**
- Create `gaia/policies/lighting.py` (replaces `lighting_basic` in `POLICY_ORDER`; keep the old module for its tests or delete it with them, the implementer's choice, stated in the report)
- Modify `gaia/household/personas.py` (new prefs with defaults)
- Test `tests/policies/test_lighting.py`

**Behaviour** (defaults are `assumption`):
- **Activity → layers:**

  | activity | layers | level |
  |---|---|---|
  | `tv` | cove + accent, main off or 0.3 | — |
  | `dinner`/`lunch`/`breakfast` in dining or kitchen | main + accent (pendant fixtures) | 0.8–1.0 |
  | `cook` | main + task | 1.0 |
  | `wind_down` | main 0.4 + cove | — |
  | `relax` | main 0.7 + accent | — |
  | `wake_up` | main 0.5 | — |
  | other awake activities | main | 1.0 |

  Applied on activity_start and on room entry while dark. Layers not wanted for the new activity are turned off on activity change (trigger `activity_end`).
- **Tunable (circadian):**
  - cct = 5000 K between 07:00 and 17:00 local; 3000 K from 19:00 until sleep; linear between.
  - `wind_down` and `sleep` prep → 2700 K.
  - Persona `cct_pref_offset_k` (default 0, range ±500).
  - CCT changes only when the light is on and the change is more than 300 K (avoids event spam).
- **RGB/strip:**
  - cove in `tv` → a warm low colour (hue 0.08, saturation 0.6, level 0.3)
  - festival class untouched (G2b)
- **Daylight switch-off:**
  - When indoor lux exceeds the persona threshold × 3 for 15 consecutive minutes, an awake occupant turns the main light off.
  - Probability per persona `daylight_off_p` (default 0.6).
  - Labels: intent `energy_saving`, trigger `daylight`.
- **Night lights:** passage and staircase `night` circuits on when anyone awake moves through between sunset and sunrise, off after 2 minutes empty (trigger `timer`).
- **Kept from G1:** exits, forgetting, sleep and cleanup, applied to every lit circuit of the room. Forgetting is decided per room, not per circuit: a forgetful person leaves all of them on.
- Persona gets `cct_pref_offset_k`, `daylight_off_p`, `dim_pref` (multiplier 0.8–1.0). Defaults keep the existing personas valid.

- [ ] Write failing tests:
  - `test_tv_uses_cove_not_full_main` (over 30 days, ≥ 70% of tv-start light intents in rooms with a cove target the cove circuit)
  - `test_cct_follows_time` (cct commands after 19:00 are ≤ 3500 K)
  - `test_daylight_off_happens` (> 0 `daylight` triggers over 60 days)
  - `test_no_cct_spam` (no two cct commands on a load within 10 min unless separated by a power cycle)
  - `test_night_light_timer`
  - All invariants still pass.
- [ ] Run → FAIL. Implement. Run the full suite → PASS.
- [ ] Commit `feat(gaia): layered, circadian and daylight-aware lighting policy`.

---

### Task 7: Climate, fan and shading policies

**Files:** create `gaia/world/season.py`, `gaia/policies/climate.py`, `gaia/policies/shading.py`; modify personas; test `tests/policies/test_climate.py`, `tests/policies/test_shading.py`.

**Behaviour:**
- `season.py`: `heat_index(city, local_date) -> float` in 0–1 from a per-city table of 12 monthly values, marked `source: assumption` (G2b replaces it with climate normals). Indian pattern: highest April–June, high July–September, mid October, low December–January. Bengaluru is flatter, Delhi has the strongest swing.
- **AC:**
  - Persona `ac_affinity` 0–1.
  - p_use = ac_affinity × heat_index for a room the person sleeps in at sleep time, or occupies for ≥ 30 min while heat_index > 0.5.
  - Setpoint = persona `ac_setpoint` (default 24, BEE default; range 20–27). Mode `cool`. Fan speed 3.
  - Off at wake (p 0.8) or on exit (subject to `forgets`).
  - Waste rule: unattended.
- **Fans:**
  - On with presence when heat_index > 0.3, speed = 1 + round(4 × heat_index).
  - On together with the AC at speed 2 (p 0.5).
  - Off on exit.
- **Exhaust:** bathroom during a bath or toilet visit (Task 9 adds toilet visits), kitchen during cook. Off 5–10 min after.
- **Shading:**
  - Curtains open at `wake_up` (position 1.0, p 0.8) or when daylight arrives with someone awake in the room.
  - Close at sunset + 0–30 min if someone is home (privacy), and at `wind_down` (p 0.9).
  - Venetian tilt 0.5 at midday when heat_index > 0.6 (labels: intent `thermal_comfort`).

- [ ] Write failing tests:
  - `test_ac_more_in_may_than_january` (Delhi, same seeds: AC on-minutes in May > 3 × January)
  - `test_setpoint_in_range`
  - `test_curtains_open_morning_close_evening` (median local open hour 5–10, median close hour 17–23)
  - `test_fan_speed_tracks_heat`
  - `test_ac_unattended_waste_recorded`
  - All invariants still pass.
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Commit `feat(gaia): climate, fan and shading policies`.

---

### Task 8: Media/AV policy

**Files:** create `gaia/policies/media.py`; modify personas; test `tests/policies/test_media.py`.

**Behaviour:**
- **TV:**
  - `tv` activity start → TV power on, source = persona `tv_source_pref` (`stb` p 0.6 if the home has `media.stb_dth`, else `streamer`), set-top box power on if used, volume = persona `volume_pref` (0.2–0.5).
  - During the activity: `pause`/`play` events at a rate of ~1 per 45 min.
  - `activity_end` → TV off. The set-top box goes to **standby** (power 0, but it counts as standby waste via `power_w.standby`; no waste interval).
- **Music:**
  - With persona `music_affinity` p, at `cook`/`dinner`/`relax`/`wind_down` start in a room with `media.speaker`/`soundbar`: power on, source `spotify` (p 0.5), `airplay` (p 0.4) or `radio`, then volume and play.
  - Stop at activity end.
- **Home theatre:** weekends and festivals at 20:00–23:00 with p 0.3: projector on, screen down (position 0 = rolled up, 1 = down: documented in YAML), AV receiver source `hdmi1`, room lights to the cove level.
- Waste rule: TV and speakers unattended. Set-top box and receiver `none` (standby).
- Labels: intent `entertainment`, device_action from the capability (`select_source`, `set_volume`, `play`, `pause`).

- [ ] Write failing tests:
  - `test_tv_session_shape` (power on → select_source → volume → … → off, in order per session)
  - `test_stb_standby_not_waste`
  - `test_music_only_with_speaker`
  - `test_source_enums_valid`
  - All invariants still pass.
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Commit `feat(gaia): media and AV policy`.

---

### Task 9: Water and utility policy; bathroom realism

**Files:**
- Create `gaia/policies/utility.py`
- Modify `gaia/behaviour/day_plan.py` (toilet visits), `gaia/policies/lighting.py` (notice rule), personas
- Test `tests/policies/test_utility.py`, `tests/behaviour/test_toilet_visits.py`

**Behaviour:**
- **Day plan:** add `toilet` to the activity enum in `labels/schema.yaml`. Insert 2–5 `toilet` activities (3–6 min, own bathroom) into awake home blocks, uniformly. Keep contiguity and ≥ 5-min blocks; when a block is too short, skip the insertion. All plan invariants still hold.
- **Notice rule** (fixes the 17 h bathroom waste):
  - Each tick, each awake member at home notices each unattended-wasted load outside their room with p = persona `notice_rate` per minute (default 0.03).
  - They switch it off (source wall, intent `noticed_waste`, trigger `notice`).
- **Geyser:**
  - On 15–30 min before `bath` (persona `geyser_lead_min`), only when heat_index < 0.7 or persona `hot_water_always`.
  - Off after the bath, with p (1 − forgetfulness).
  - Waste rule `overrun:60`.
- **Pump:** once or twice a day (morning 06:00–08:00, evening p 0.4), 20–40 min, source `system`, intent `utility`, trigger `timer`.
- **Chimney:** with `cook`, speed 2–3, off at the end.
- **RO purifier:** short `utility` runs after cooking (system).
- **Washing machine:** 2–4 times a week, by the homemaker or on weekends, intent `utility`.

- [ ] Write failing tests:
  - `test_toilet_visits_inserted` (2–5 per awake day; plan invariants hold)
  - `test_bathroom_waste_median_under_2h` (60 days × 20 homes: the median bathroom waste interval is < 120 min)
  - `test_geyser_before_bath`
  - `test_geyser_overrun_waste`
  - `test_pump_daily`
  - All invariants still pass.
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Commit `feat(gaia): utility policy, toilet visits and noticing wasted loads`.

---

### Task 10: Sensors and sensor richness

**Files:** create `gaia/policies/sensors.py`; modify `gaia/home/inventory.py` (richness); test `tests/policies/test_sensors.py`.

**Behaviour:**
- Home parameter `sensor_richness` ∈ `none` (p 0.4), `some` (0.4), `rich` (0.2), stored in the manifest.
  - `some`: motion sensors in passages and bathrooms, and a contact sensor on the main door.
  - `rich`: presence (mmWave) in living and bedrooms, motion everywhere else, a lux sensor in living, temperature and humidity per bedroom, an energy meter on the home.
- Sensor events use `kind=sensor` and `source=system`. Labels: intent `system`, trigger `presence_tick`, device_action `none` (add `none` to the device_action enum in `labels/schema.yaml`).
  - **Motion:** `occupancy` 1 on entry. While moving, re-trigger every 2–5 min for awake occupants. 0 after a 2-min hold with nobody awake in the room. Sleepers do not trigger motion.
  - **Presence:** 1 while anyone is in the room (asleep included), 0 when empty, 30 s hold.
  - **Contact:** main door open/close when someone leaves or returns from away.
  - **Lux:** a reading every 15 min (indoor lux plus lights-on contribution: +150 per lit main circuit, `assumption`).
  - **Temperature:** every 15 min, from a toy model (outdoor heat_index → 24–36 °C; AC on pulls it towards the setpoint over 20 min).
  - **Energy meter:** total W every 5 min = Σ powered loads' `power_w` by state.
- Sensor events never get a waste label, and lights never react to sensors (MSI will).

- [ ] Write failing tests:
  - `test_motion_consistent_with_presence` (every motion=1 event's room has an awake member that minute)
  - `test_no_motion_from_sleepers`
  - `test_presence_covers_sleep`
  - `test_energy_meter_matches_sum`
  - `test_richness_none_has_no_sensor_events`
  - All invariants still pass.
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Commit `feat(gaia): sensor policy with richness levels`.

---

### Task 11: ETS layout importer (anonymised)

**Files:**
- Create `gaia/ets/__init__.py`, `gaia/ets/importer.py`, `gaia/ets/maps.yaml`, `scripts/import_ets.py`
- Modify `gaia/home/inventory.py` (calibration hook)
- Test `tests/ets/test_importer.py`, `tests/ets/fixtures/sample_KNXdata.json` (hand-made, invented names, including the sentinel `CLIENTSECRET`)

**Interfaces:**
- `maps.yaml`:
  - Room-name keywords → room_type, e.g. `master|mbr|bed|cbr|gbr → bedroom`, `living|hall|lounge|drawing → living`, `dining → dining`, `kitchen|dry kitchen → kitchen`, `pooja|puja|prayer → pooja`, `bath|toilet|powder|wash → bathroom`, `theatre|theater|ht → home_theatre`, `study|office|library → study`, `gym → gym`, `passage|corridor|lobby|stair|steps → passage`, `terrace|balcony|deck → terrace`, `garden|landscape|facade|parking|gate → outdoor`, `dressing → dressing`, `dh|maid|staff|utility → utility`.
  - Load `ldTyp` + name keywords → class + fixture + circuit:
    - Dimmer → `light.dimmable`; Tunable → `light.tunable`; Curtain → `shade.curtain`; HVAC → `climate.split_ac`; Scene → `keypad.scene`
    - Switch + `fan` → `fan.ceiling`; Switch + `geyser` → `water.geyser`; Switch + `exhaust` → `fan.exhaust`; Switch + `socket|plug` → `plug.smart`; other Switch → `light.onoff`
    - Name keywords for fixtures: `down` → downlight, `profile`, `cove`, `pendant|chandelier`, `strip`, `step|foot`, `spot|track`, `wall wash`
  - Unmatched → `unknown`, counted in the report.
- `import_knxdata(path: Path, project_id: str) -> EtsLayout` with `EtsLayout(project_id, rooms: tuple[(room_type, tuple[(class, fixture, circuit)])], stats)`.
  - **No names are kept.** `project_id` is `sha256(file bytes)[:12]`.
- `is_residential(layout) -> bool`: ≥ 4 rooms mapped to residential types, ≥ 60% of loads in mapped rooms, and no single room holding more than 50% of loads.
- `scripts/import_ets.py <knxdata_dir> <out_dir>` writes `out_dir/<project_id>.json` layouts plus `summary.json` (counts only). Default out dir: `data/ets_layouts/` (git-ignored).
- **Archetype `ets`:** `build_home("ets", …)` picks a residential layout (rng) from `GAIA_ETS_DIR` (env var or config `ets_dir`) and builds rooms and loads from it (room names `<type>_<n>`, protocols by `knx_dali`). Missing dir → a clear `ValueError` naming the setting. `RunConfig.archetypes` may include `ets` only when `ets_dir` is set.
- **Calibration:** `inventory_stats(layouts) -> dict[room_type, {class: mean count}]`, printed by the script for review. Not auto-applied in G2a.

- [ ] Write failing tests:
  - `test_room_mapping` (fixture rooms map as expected)
  - `test_load_mapping`
  - `test_no_names_in_output` (Review Focus 5: the serialised layout and summary never contain `CLIENTSECRET` or any fixture room or load name)
  - `test_function_grouped_project_not_residential`
  - `test_ets_archetype_builds_home` (from the fixture layout)
- [ ] Run → FAIL. Implement. Run → PASS.
- [ ] Run the script locally on the real KNXdata in `/tmp/claude-1000/-home-iris-Miantic-MSI/8b4a6833-3ed6-40d3-a94b-5035c41b401a/scratchpad/ets/` (already converted) into `data/ets_layouts/`. Report counts only: residential layouts, unknown-load share. **Do not commit any output.**
- [ ] Commit `feat(gaia): anonymised ETS layout importer and ets archetype`.

---

### Task 12: Dataset checker and quality report; spec sync

**Files:**
- Create `gaia/quality/__init__.py`, `gaia/quality/check.py`
- Modify `gaia/cli.py` (`check` subcommand), `docs/plans/gaia-spec.md` (§4.1 protocol enum, §4.2 labels v1 replacing gt_cause, §8 label sources; minimal edits)
- Test `tests/quality/test_check.py`

**Interfaces:** `check_dataset(root: Path) -> dict` (also `uv run python -m gaia check <root>`, which prints JSON and exits 1 on any violation). It reads shards and computes:
- **Violations** (must be 0):
  - duplicate event_ids
  - non-monotonic per-load states
  - a state without a command
  - a command without exactly 1 label
  - a label referencing a missing event
  - an enum outside the label schema
  - a value outside the registry range
  - a waste interval on an off load
  - presence overlap per person
  - motion from a sleeper
  - a non-empty home with zero events
- **Stats:**
  - events per home-day
  - per class / per domain / per protocol share
  - label coverage per level (share ≠ `unknown`/`none`)
  - intent distribution
  - waste minutes per home-day by class
  - median waste interval by room type
- `generate` writes `check.json` next to the world dir after generation. A run with violations exits non-zero.

- [ ] Write failing tests:
  - `test_clean_world_passes` (generate homes=6 across all archetypes except ets, days=7 → violations all 0)
  - `test_detects_injected_violation` (copy a shard, delete one label row → the checker reports it)
  - `test_cli_check_exit_code`
- [ ] Run → FAIL. Implement, and update the spec sections. Run the full suite → PASS.
- [ ] Run `uv run python -m gaia generate configs/g1-smoke.yaml`, then `uv run python -m gaia check data/g1-smoke`. Report the stats summary (events per home-day, class count, label coverage, median bathroom waste) and `scripts/bench_g1.py` seconds per home-year.
- [ ] Commit `feat(gaia): dataset checker and quality report; spec labels v1`.

---

## Done means (G2a exit)
- Full suite green. `gaia check` reports 0 violations on a mixed run (7 archetypes, 30 days).
- Generated homes cover ≥ 40 distinct load classes across a 50-home run. Label coverage of intent and device_action ≥ 99% (not `unknown`).
- Median bathroom waste interval < 2 h.
- Speed measured. Target: < 10 s per home-year for a villa. If missed, report it; don't optimise here.
- Not in G2a: TUS calibration, weather, festivals, power cuts, scenes and schedules (keypad events), drift, faults, thermal and energy tables (G2b); public dataset converters (G2c).
