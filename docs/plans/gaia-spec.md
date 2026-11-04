# GAIA: the MSI home-world simulator and dataset generator (spec v0.1)

Date: 2026-11-03 · Author: Nitin.J · Status: **approved 2026-11-03**
Parent: `docs/research/msi-research-roadmap.md` (Phase 0 and Phase 1). agres: P19, D15–D18, D21, N20.

> GAIA, the Greek earth mother who gives birth to worlds, keeps the OKAS naming line (hermes, hades, callisto…).
> GAIA builds thousands of homes, fills them with people, lets them live for months,
> and records everything their homes would see.

---

## 1. Why GAIA exists

MSI has to learn habits in homes it has never seen. Real data is scarce: one office site, 20 days, and
logs with no source or room (N20). No public dataset is protocol-agnostic or actuator-centric, and none is Indian.
So the dataset is the product. GAIA is the factory.

### Goals
1. Generate **realistic, time-accurate event logs** for any number of homes: *x* load layouts × *N* homes × *D* days → *z* events.
2. Emit the **same MSI Event Schema** that the real on-board logger will write, so synthetic and real data mix without conversion.
3. Record **ground truth that real logs never have**: who did it, why, which activity, which routine, which events were forgotten.
   This gives free supervision for routine discovery, intent prediction and energy-waste detection.
4. **Indian homes first**: climate, seasons, festivals, power cuts, domestic help, joint families. Building types are pluggable (offices later, D21).
5. **Protocol-agnostic**: KNX, Matter, Zigbee and others appear only as profiles that add quirks such as latency, echoes and drops (D15).
6. **Calibrated to reality** with real statistics (India Time Use Survey 2024, real OKAS logs, sun position, climate normals),
   and **proven** by a fidelity report before any model trains on it.
7. **Closed-loop mode**: MSI can act *inside* GAIA and the simulated people react by accepting, declining, undoing or ignoring.
   The same engine serves as dataset factory and as test arena (Phases 4–5).
8. **Reproducible**: every home is a pure function of (world version, seed).

### Non-goals (v0.1)
- No full building physics (EnergyPlus). A first-order thermal model per room is enough for AC behaviour and energy estimates.
- No video, audio or 3D rendering.
- No LLM writing events directly. LLMs may write *configurations* offline (personas, life events) that the deterministic engine runs.
- No on-board code. GAIA runs on a PC or server. The on-board logger is a separate Phase 0 deliverable that shares the schema.
- No offices yet. Only the archetype plug-in point exists.

---

## 2. The core idea: simulate causes, not logs

A log line is the *last* step of a chain. GAIA simulates the whole chain, so the statistics come out
right without hand-tuning:

```
World (date, sun, weather, power, festival)
  └─> Person (persona, chronotype, mood, plan for today)
        └─> Activity (wake, puja, cook, work, TV, sleep...) in a Room
              └─> Need (too dark, too hot, want privacy, leaving, ambience)
                    └─> Interaction (wall switch | app | voice | HomeKit | scene | schedule | nobody: forgot)
                          └─> Device event (abstract) ──> Protocol profile (quirks) ──> MSI Event log
```

Patterns real people show come out on their own:
- Lights follow **sunset, not the clock**: the "need light" threshold crosses as lux falls.
- **Forgotten loads** happen because a person leaves a room with a forgetfulness probability.
- Routines **drift** because life events change the plan generator.
- Several occupants **conflict**: one turns the AC to 22 °C, another to 26 °C.
- Power cuts cause the "all-off burst then restore storm" we already see in the real log (`KNX interface DISCONNECTED` → every load `Turned-Off`).

---

## 3. Architecture

GAIA is a Python package (`gaia/`) with one engine and pluggable layers. Each layer has one job, a small
interface, and its own tests.

| # | Layer | Job | Main inputs | Output |
|---|---|---|---|---|
| 1 | **World** | calendar, sun, weather, grid, tariff | city, date range, seed | `WorldTimeline` (per-minute context) |
| 2 | **Home** | build a home: archetype → rooms → loads → protocol per load → optional sensors | archetype plug-in, load-class registry | `HomeManifest` |
| 3 | **Household** | who lives here: members, personas, preferences, schedules | persona library, TUS calibration | `Household` |
| 4 | **Behaviour** | each person's day: plan → activities → room presence → needs | persona, world, life-event state | person-state stream |
| 5 | **Interaction** | turn a need into an action through a chosen interface, or forget it | tech-affinity, room UI | intended device commands |
| 6 | **Automation** | the home's own scenes and schedules: fire them, and users edit them over time | household habits | scene/schedule events |
| 7 | **Device and protocol** | apply the command to device state; protocol profile adds latency, echoes, drops, reconnects | load registry, protocol profiles | raw device events |
| 8 | **Faults** | power cuts, gateway drops, stuck sensors, clock skew, duplicate events | fault profile | injected anomalies |
| 9 | **Life events and drift** | seasons, travel, guests, exams, new baby, new job, festival, illness, device added or renamed | life-event library | plan modifiers over time |
| 10 | **MSI response** (closed loop) | how each person answers MSI proposals and actions: accept, decline, undo, ignore | latent preferences, trust state | feedback events |
| 11 | **Recorder** | write the MSI Event Schema plus context plus ground truth to Parquet | all of the above | dataset shards |

**Engine**: a per-home discrete-event simulator (priority queue of timestamped events, plain `heapq`,
no framework). Time resolution is milliseconds, with a 1-minute context tick for world and thermal updates.
Homes are independent, so generation parallelises across processes with no shared state.

**Determinism**: one `numpy.random.Generator` per (home seed, layer). Adding a layer never shifts the
random streams of the other layers.

### 3.1 World layer (India first)
- **Sun**: real solar position computed astronomically for the city's latitude and longitude, giving sunrise, sunset and outdoor lux. Not sampled.
- **Climate**: per-city monthly normals (temperature, humidity, cloud) with day-to-day weather noise. Cities are presets
  (Hyderabad, Bengaluru, Chennai, Mumbai, Delhi, Kolkata, Pune), one per climate type. The normals source is
  an open question (§11).
- **Seasons**: summer (Mar–Jun), monsoon (Jun–Sep), post-monsoon, winter. They drive AC, fan and geyser use and curtain behaviour.
- **Calendar**: weekends, national and regional festivals (Diwali, Holi, Navratri, Ganesh Chaturthi, Pongal and Onam by
  region, Eid, Christmas), school terms, exam season, summer vacation.
- **Grid**: power-cut process per city tier (frequency and duration); inverter or DG present per home; tariff slabs and time-of-use.

### 3.2 Home layer
- **Archetype plug-ins**: `apartment_1bhk`, `apartment_2bhk`, `apartment_3bhk`, `independent_house`, `villa`.
  Each defines the room graph (adjacency for movement) and probabilities over load inventories per room.
  Office and clinic archetypes plug into the same interface later.
- **Load-class registry** (protocol-free, D15). Each class declares capabilities, rated power, and a state machine:
  `light.onoff`, `light.dim`, `light.cct`, `light.rgb`, `fan.speed`, `ac` (power, mode, setpoint, fan),
  `curtain.position`, `blind.tilt`, `plug`, `geyser`, `water_pump`, `exhaust`, `tv_media`, `mosquito_repellent`, `scene_button`.
  Sensors: `motion`, `door`, `temp`, `humidity`, `lux`, `energy_meter`.
  A new v2 feature is a new registry entry, never an engine change.
- **Sensor richness** is a home parameter: *none*, *some* or *rich*. We don't yet know what real installs have
  (Iris, Q2), so GAIA generates all three. That forces MSI to work even with only actuator events.
- **Protocol per load**: KNX, Matter, Zigbee or Wi-Fi, mixed within one home, each with a profile (§3.7).

### 3.3 Household layer
- **Persona library** for Indian homes: working professional (office, hybrid, WFH, night shift), homemaker, college student,
  school child, elderly parent, domestic help (visits at fixed hours), cook, driver, guest.
- Each persona has: chronotype (wake and sleep distributions), daily activity templates, comfort preferences
  (AC setpoint, preferred brightness and colour temperature, fan speed), **tech affinity** (wall switch vs app vs voice),
  forgetfulness, energy consciousness, religiosity (morning and evening puja lamp), TV habits, and its **latent routine set**
  (the ground-truth routines MSI should discover).
- Households are compositions: nuclear, joint, single, couple, roommates, elderly alone.
- **Calibration**: activity timings (wake, sleep, cooking, work, leisure, religious practice) are fitted to the
  **India Time Use Survey 2024** microdata (MoSPI/NSO: about 10.2M activity rows, clock-timed, urban/rural; microdata.gov.in catalog 236).
  We use urban rows by state as priors for persona schedules. This anchors Indian daily rhythms to national data, not guesses.

### 3.4 Behaviour layer
- Each morning, every person samples a **day plan** from their persona templates, conditioned on the day type
  (workday, weekend, festival, holiday, sick, travel), with time jitter.
- Activities place people in rooms. Movement follows the room graph.
- **Needs** are evaluated each minute per occupied room: light need (indoor lux from outdoor lux, curtains and lamps against the person's threshold),
  thermal need (room temperature from a first-order RC model, §3.9, against personal comfort bounded by IMAC),
  privacy (curtains at night), ambience (a scene for TV or dinner), and leaving (switch off, or forget).

### 3.5 Interaction layer
- Turns a need into an action. The interface is chosen by tech affinity, which device is near, and context
  (voice while cooking, app from bed, wall switch at the door).
- **Forgetting**: on leaving, each active load is switched off with p = 1 − forgetfulness, adjusted by energy consciousness and hurry.
  Forgotten loads are labelled in ground truth: free labels for waste detection.
- **Habit compression**: a person who repeats a manual sequence often enough may *create a scene or schedule* (§3.6).
  That is the human version of what MSI will later propose.

### 3.6 Automation layer
- Homes start with 0 to N scenes and schedules (installer defaults plus user-made).
- Scenes fire as one source event plus member events. Schedules fire by clock or sun offset.
- Users **edit** automations over time (change a time, add a load, disable one), driven by drift (§3.8).
  This matters because MSI must learn to *edit* automations, not just create them (D16).

### 3.7 Device and protocol layer
- Device state machines enforce physics: no dimming an off light without turning it on, curtains have travel time, AC compressor min-off time.
- **Protocol profiles** add the quirks a real gateway shows: command→state latency distribution, state echo or
  duplicate, dropped telegrams, reconnect bursts, polling and status-read storms at startup (as in the real hermes log).
  The model must learn to see *through* protocol noise to human intent.

### 3.8 Life events and drift
A library of time-bounded modifiers on the plan generator. Some examples:
- *Seasonal*: summer brings AC at night, monsoon brings lights earlier and closed curtains, winter brings the geyser in the North.
- *Episodic*: guests for a week (new persona, guest-room usage), hometown trip (home empty 2–3 weeks; maybe only the domestic help),
  exam season (late-night study lights), festival (decorative lights, everyone awake late), illness (one person home all day).
- *Structural*: new job (shifts wake time permanently), new baby (night activity), a member moves out, a new load installed,
  a load renamed or moved, a bulb dies (commands but no change), a device replaced with another protocol.
- Every drift event is logged in ground truth with its start and end, so drift-detection benchmarks come for free.

### 3.9 Thermal and energy (deliberately simple)
- A first-order RC model per room: outdoor temperature, solar gain through curtains, occupants, AC cooling capacity and setpoint.
- Energy per load = rated power × on-time × level factor (dim level, fan speed, AC duty cycle from the RC model).
  It is labelled as an **estimate**, as the research recommends. It is good enough to rank energy-saving proposals and compare MSI on vs off.

### 3.10 MSI response model (closed loop)
- MSI's outputs are **structured proposals and actions** (D17): `create_scene`, `edit_scene`, `create_schedule`,
  `edit_schedule`, `activate`, `turn_off_unused`, `setpoint_adjust`, and more. The **proposal catalog** grows over time.
- Each person answers based on: how well the proposal matches their *latent* routines and preferences, current activity
  (no proposals at 2 a.m.), a **trust state** (rises with good proposals, falls with bad ones or with nagging), and an annoyance budget.
- For autonomous actions, people may **undo** within N minutes, override later, or not notice.
  Exposure is recorded: whether they saw it. The research shows unseen actions must not count as rejections.
- This gives Phases 4–5 a safe sandbox. MSI can make mistakes in a million simulated homes before it touches one real one.

---

## 4. Data model

### 4.1 MSI Event Schema v0 (shared by GAIA and the real on-board logger)

| field | type | notes |
|---|---|---|
| `ts_utc` | int64 ms | event time |
| `tz_offset_min` | int16 | local time is derived (India +330) |
| `home_id` | string | opaque |
| `event_id` | string | unique within home |
| `kind` | enum | `command`, `state`, `scene_fired`, `schedule_fired`, `automation_edit`, `sensor`, `system`, `msi_proposal`, `msi_action`, `user_feedback` |
| `room_id` / `room_type` | string / enum | room type is abstract (`bedroom`, `living`, `kitchen`, `puja`, …) |
| `load_id` / `load_class` | string / enum | from the load-class registry |
| `capability` | enum | `power`, `level`, `cct`, `position`, `setpoint`, `mode`, `fan_speed`, … |
| `value`, `prev_value` | string | normalised (0–1 for levels, °C for setpoint). Stored as strings: floats via Python `repr(float)`, e.g. `"1.0"` |
| `source` | enum | `wall`, `app`, `voice`, `homekit`, `scene`, `schedule`, `msi`, `system`, `unknown` |
| `actor` | string | user id or role if known, else `unknown` |
| `protocol` | enum | `knx`, `matter`, `zigbee`, `wifi`, `other`. Never used as a model feature by default. |
| `ref_id` | string | links an echo to its command, a scene member to its scene, feedback to a proposal |
| `domain` | enum | `synthetic` / `real` / `public` (for audit) |

The context stream (per home, per minute): outdoor temperature and humidity, outdoor lux, sun altitude, weather, grid up or down,
tariff band, plus any real sensors present.

### 4.2 Ground-truth tables (synthetic only, never available on the board)
- `gt_cause` per event: cause, person, activity. G1 causes are `need_light`, `leave_room`, `sleep` and `forgot_cleanup`; the others (`need`, `routine:<id>`, `scene`, `schedule`, `forgot`, `accident`, `guest`, `help_staff`, `msi`, `protocol_noise`) arrive in G2.
- `gt_waste`: `load_id`, `person_id`, `start_ts_utc`, `end_ts_utc` (null if open at the end of the run). A main light is on while no awake member is in its room. Intervals shorter than 60 s are not emitted.
- `gt_presence`: `person_id`, `room_id` (null = away), `start_ts_utc`, `end_ts_utc`, `asleep`. Maximal runs of constant room and sleep state per person, tiling the whole run.
- `gt_routines`: each person's latent routines (the answer key for routine discovery).
- `gt_drift`: life-event intervals.
- `gt_energy`: per-load energy estimates.

The model must never need these at inference. They are for auxiliary training targets and evaluation only.

### 4.3 Files
Parquet per shard (`world=<ver>/split=<s>/home=<id>/events.parquet`, plus `context`, `gt_*`), a `manifest.json` per
home (archetype, household, loads, seeds), and a `dataset_card.md` per release (version, sizes, fidelity scores, known gaps).

---

## 5. Making it real: calibration and fidelity

1. **Priors from national data**: India TUS 2024 for activity timing. Sun from astronomy. Climate from city normals.
2. **Priors from OKAS**: real `loadData.json` and `rooms.json` shapes for load inventories. Real logs (once structured, N20) for
   event rates, inter-event gaps and protocol quirks.
3. **Public behaviour datasets** (CASAS etc., licences permitting): sequence statistics for cross-checks.
4. **Fidelity report** per release, comparing synthetic and real:
   events per home-day, per-load on-time histograms, inter-event gap distribution, time-of-day profiles, scene and schedule share,
   and a **discriminator test** (a classifier trying to tell real from synthetic; a lower AUC means more realistic).
   Plus **train on synthetic, test on real** once real logs exist.
5. **Gate**: a release can feed model training only if its fidelity scores pass the thresholds set in GAIA milestone G3.

## 6. Mixing real and synthetic: "home twins"

Real data is precious but thin, so GAIA grows it:
- **Domain tags** on every row (`domain`) so any training mix can be audited.
- **Home twins (inverse simulation)**: fit GAIA's household parameters to one real home's log (wake times, routines,
  forgetfulness, interface mix). Then sample **siblings**: the same home with jittered timings, a different season,
  a guest week, a new member. One real home becomes a hundred plausible variants that stay anchored in reality.
- **Real anchors in splits**: real homes are always held out for testing first. They are the final judge.

## 7. Scale and reproducibility

- Config (YAML, validated with pydantic): `homes`, archetype mix, city mix, household mix, `days`, sensor richness mix,
  protocol mix, fault profile, life-event rates, seed, world version.
  This is Iris's "x loads in x^n houses → z logs", made explicit.
- Shard by home and generate in parallel (multiprocessing). Target to measure in G1, not a promise:
  **1 home-year in under 10 s on one core**, so 10k home-years in about 1 hour on 8 cores.
- Same config plus same world version produces byte-identical output, checked by a test.

## 8. What the dataset is for: tasks and splits ("MSI-Bench")

| Task | Label source |
|---|---|
| next event (what) | events |
| time to next event (when) | events |
| routine discovery | `gt_routines` |
| intent and cause classification | `gt_cause` |
| forgotten or wasted load detection | `gt_waste` |
| drift detection | `gt_drift` |
| proposal acceptance | MSI response layer |
| cold start (first 1/3/7/14 days) | time split |

Splits: **leave-homes-out** (the main one), future-time, unseen archetype, unseen city/climate, unseen protocol mix, sensor-poor only.

## 9. Extensibility points (things that must never require an engine change)
Archetypes (offices later), load classes (future v2 features), personas, life events, protocol profiles, fault profiles,
proposal types, cities. Each is a registered plug-in with a schema and a unit test.

## 10. Testing
- **Determinism**: same seed produces identical shards.
- **Invariants** (property tests over many seeds): no state change without a cause; state machines never illegal;
  everything restores after a power cut; no person in two rooms; scene member events follow the scene event; timestamps monotonic per home.
- **Statistical tests**: persona timing distributions match their TUS-calibrated targets within tolerance.
- **Golden homes**: a few hand-checked homes with known routines; their ground truth must be recoverable by a simple miner.
- **Fidelity report** as a release gate (§5).

## 11. Milestones inside GAIA
- **G0, schema**: MSI Event Schema v0 frozen and shared with the on-board logger spec. Load-class registry v0.
- **G1, walking skeleton**: World (sun, calendar) + one archetype + two personas + Interaction + Recorder. One home-month, deterministic, invariants pass. Throughput measured.
- **G2, Indian breadth**: all archetypes, persona library, TUS calibration, scenes and schedules, drift library, faults, protocol profiles, thermal and energy.
- **G3, fidelity**: fidelity report against the real OKAS logs (structured logger from Phase 0) and TUS. Thresholds set. First dataset release `gaia-world-1`.
- **G4, closed loop**: MSI response model and a gym-style API for Phases 4–5.

## 12. Risks
1. **Self-fulfilling realism**: the model learns GAIA's rules, not people. Mitigations: TUS calibration, discriminator tests, home twins, real-only test sets.
2. **Too much detail too early**: the layers invite gold-plating. G1 is deliberately tiny.
3. **TUS fit**: the survey records activities, not device use. Mapping activity to need to device is our modelling assumption and must be validated against real logs.
4. **Feedback loops in closed-loop mode**: MSI-caused events are tagged `source=msi` and excluded from habit learning by default.

## 13. Open questions (non-blocking; defaults chosen)
- Climate normals source for the city presets. Default: an open climate-normals dataset, to be verified in G2.
- India TUS 2024 licence and terms of use for derived priors. Must check before G2.
- Real logger rollout on the v1.2 dev board: a separate Phase 0 spec (`docs/plans/msi-logger-spec.md`) that reuses §4.1.
- Technology: Python 3.12, numpy, pyarrow, pydantic, pytest. GAIA runs off-board, so there is no RAM constraint here.
