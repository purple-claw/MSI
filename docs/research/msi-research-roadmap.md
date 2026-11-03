# MSI — Research Roadmap (v0.1)

Miantic Super Intelligence: on-device, self-adapting automation for OKAS HomeKit v2.
Date: 2026-11-03. Status: research synthesis, not a spec. Nothing here is approved for build yet.

Inputs: five research streams in `streams/` (raw reports, mostly snippet-level),
plus three claims I re-checked by search (marked **[checked]**). Anything else from the streams
should be treated as a lead, not a fact, until we read the primary source.

---

## 1. What we are building (agreed so far)

- One compact **System-1 base model**: fast, habit-driven decisions with no step-by-step reasoning.
  Pre-trained offline on many homes (simulated plus public datasets) and shipped the same to every board (D9).
- **Per-home adaptation on the board**: base frozen; only small adapter weights train on the board's CPU from that home's own logs.
- Optional **System-2**: a slow pass, for example nightly, that reviews habits, retires stale ones, and recalibrates confidence.
- **Hard constraint**: ARM board with 2–4 GB RAM and an NPU. Design for 2 GB. MSI shares the board with the OKAS services (P8).
- **Hardware not final** for v2 (N11). Keep the NPU behind a runtime abstraction.
- Not a chatbot. No maths, no code generation. Collect → find patterns → decide → act, inside strict guardrails.

## 2. What the research says (the findings that shape the design)

### 2.1 The model already has a close precedent
- **DomusFM** (Fiori, Civitarese, Salim, Bettini, 2026) is a foundation model pre-trained on smart-home
  sensor *events*. It describes events by abstract attributes (type, status, timing) rather than device IDs,
  so one model works across homes with different devices. It was tested leave-one-dataset-out on 7 public
  datasets, for activity recognition and next-k event prediction. **[checked]** exists, with these authors and this design.
  Aggregators report about **36M params** and about **9.6 ms per 30-event window on a CPU gateway**.
  Those two numbers are not yet checked against the PDF. https://arxiv.org/abs/2602.01910
  → This sits inside our 10–100M target and validates the "abstract event tokens, not device IDs" idea.
  It is exactly what lets one base model ship to 200 different homes.
- **SmartUDI / DeepUDI** (Jingyu Xiao, Qing Li et al., 2023) predict user–device interactions from routine,
  intent and multi-level periodicity, and are described as deployable on edge nodes. **[checked]**
  https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1634.pdf, DOI 10.1145/3610906
- **SmartSense**: self-attention over (device, control, day-of-week, hour). https://arxiv.org/pdf/2208.06089
- **Timing Matters**: predicts *when* the next action happens, not just what. https://arxiv.org/abs/2411.18719
- LLM agents (SAGE, Sasha) are cloud-scale and **don't belong on the board**. At most an LLM is an
  offline teacher or simulator.

### 2.2 Data is the real bottleneck
- Public datasets: CASAS (many homes, motion and door sensors), ARAS, Orange4Home, MARBLE, ContextAct@A4H, MuRAL.
  They are good for pre-training *behaviour*, but **none are KNX or actuator-centric** (they log sensors more than switch, dimmer and HVAC commands).
  Licenses are unverified for all of them.
- No public KNX dataset and no KNX simulator exist. **We have to generate our own and start collecting real logs now.**
- Simulators to build on: LoadProfileGenerator (MIT license confirmed, agent-based residential load profiles),
  Sinergym/EnergyPlus (HVAC physics), OpenSHS/SIMADL (activity events), and LLM-persona household simulators (2025–26 prototypes).
- Validation methods exist: train on synthetic and test on real, compare distributions (on-time, inter-event gaps, daily energy),
  and train a real-vs-synthetic discriminator.

### 2.3 On-device learning is feasible, with one trap
- The trap (TinyTL): **training memory is dominated by activations, not parameters**. Small adapters
  are not automatically cheap. Use batch size 1, short event windows, gradient checkpointing, and measure peak RSS.
- NPUs on these boards are inference-only (assumed, not confirmed per vendor). **Adapter training runs on the CPU**, at night.
- Forgetting: a small per-home replay reservoir plus adapter snapshots per period is cheaper than EWC. Use EWC on adapters only if needed.
- On-device training frameworks are immature on aarch64 Linux. ExecuTorch can do LoRA backward with custom
  work; ONNX Runtime training and TFLite on-device training are unverified. **This is our biggest technical unknown.**
- Federated learning (sharing only quantized adapters, FLoCoRA-style, via Flower) is a later phase, after a privacy review.

### 2.4 Guardrails: there is a known layered pattern
- **Shielding** (Alshiekh et al., AAAI 2018): a fixed safety layer that removes or replaces unsafe actions before they run.
- **IoTGuard / IoTSafe** (NDSS 2019 and 2021): runtime policy enforcement for smart homes.
- Confidence gating: **conformal prediction** gives a principled "abstain" signal.
  Parasuraman's levels of automation give the ladder: suggest → ask → act → act silently.
- Learning from overrides: an undo within N minutes is a negative signal, **but only for actions the user actually saw**.
  There is no good published smart-home study of this, so it is open research and a contribution we can make.
- Feedback-loop risk (Sculley 2015): MSI's own actions must be quarantined from its training data,
  or it learns from itself.
- Privacy: occupancy leaks easily. Keep logs on the board.

### 2.5 Energy: modest, real, and evidence-ranked
Best evidence first (study-specific numbers, not promises):
1. Lighting: occupancy and off-delay control, about 20–26% of lighting energy in offices.
2. HVAC setback when unoccupied: about 5–21% in residential studies.
3. Standby and forgotten loads: about 4–26% of household electricity across studies.
4. AC setpoint (India, BEE): about 6% per °C, with a 24 °C default.
5. Curtains and solar gain: mixed field evidence. Treat as site-specific.
6. Independent learning-thermostat studies show only about 5%. Vendor "20%+" claims are not independent.

Comfort floors must be hard limits: **IMAC** for India (neutral 21.5–28.7 °C for outdoor 13–38.5 °C),
**ASHRAE 55 adaptive** elsewhere.
Measure savings as paired counterfactuals in the simulator (same occupant trace with MSI on and off) and as A/B
in the field. Rated wattage × on-time is only a labelled estimate.

### 2.6 Hardware: three serious candidates
| | NPU | Strengths | Risks |
|---|---|---|---|
| **Rockchip RK3576** | 6 TOPS (sparse figure; dense unverified) | best transformer toolkit work (SDPA, Flash Attention, LayerNorm in RKNN-Toolkit2); mainline driver arriving | mainline inference unproven on this chip |
| **NXP i.MX 95** | ~2 TOPS (2023 press) | 10–15 year longevity program; industrial support | weak and unverified NPU; closed op coverage |
| **MediaTek Genio 510** | 3.2 TOPS | LiteRT and ExecuTorch NPU backends; Yocto | closed SDK; price and power unknown |

- The Rockchip mainline NPU driver ("Rocket", `CONFIG_DRM_ACCEL_ROCKET`) is in **Linux 6.18+**, with Mesa 25.3 userspace. **[checked]**, via LKDDB and Radxa forum.
  Our v1.2 board already runs kernel 6.18.
- **ExecuTorch** has the broadest embedded NPU backend list (Ethos-U, NXP, MediaTek, Qualcomm, with XNNPACK CPU fallback),
  so it is the most hardware-agnostic runtime choice. ONNX Runtime has a Rockchip EP (preview).
- The decisive test is not TOPS. It is **whether our model's operators map to the NPU in INT8**.
  Choose a backbone with standard ops (attention/MatMul, LayerNorm, conv/TCN, GRU). Avoid custom SSM/Mamba scan ops until proven.

## 3. Reference architecture (direction only, to be specced)

```
KNX bus ─ callisto ─ okas/knx/state ─┐
app/HomeKit/Alexa ─ hermes (user cmds)┤
schedules/scenes ─ chronos ───────────┤
                                      ▼
                         ┌── MSI service (new package) ──────────────────────────┐
                         │ 1 Event logger   → local event store (tagged real/MSI) │
                         │ 2 Tokenizer      → abstract events (type,status,time,  │
                         │                    room, context)  [DomusFM-style]    │
                         │ 3 System-1 model → next action + timing + confidence  │
                         │    (frozen base on NPU + per-home adapters)           │
                         │ 4 Guardrail stack L0–L6 (deterministic, never learned)│
                         │ 5 Actuator       → same command path as the app       │
                         │ 6 Feedback       → exposure + undo/override labels    │
                         │ 7 Nightly        → adapter training (CPU), System-2    │
                         └───────────────────────────────────────────────────────┘
```

Guardrail stack (from stream 4, adapted):
- **L0 hard envelope**: setpoint min/max, curtain travel, load whitelist, never touch safety loads. Deterministic.
- **L1 policy shield**: rules over state, such as no heating or cooling while away or a window is open, and no conflict with active scenes.
- **L2 rate limits**: per-device cooldown, hourly action budget, quiet hours.
- **L3 confidence gate**: conformal or selective threshold → suggest / ask / act-with-reason / act silently.
- **L4 occupant arbitration**: never override a manual change within a hold period.
- **L5 feedback hygiene**: only exposed actions count as negatives; MSI's own actions quarantined; kill switch.
- **L6 sensor sanity and explanation**: reject on stale or implausible state; every action gets a "why" record.

## 4. Phase-by-phase roadmap

Each phase ends at a **gate** (evidence, not a date).

### Phase 0: Foundations and data capture (start now; this unlocks everything)
- Freeze the **MSI event schema**: command, state change, source (app, HomeKit, Alexa, schedule, scene, wall switch, MSI), room, load type, context.
- Add a **logging-only MSI agent** to the v1.2 dev board (and any willing site) to start collecting real KNX usage. Read-only; it never actuates.
- Check dataset licenses (CASAS, ARAS, …). Read DomusFM and SmartUDI in full.
- Reproduce one baseline (SmartSense- or DomusFM-style) on CASAS on a PC.
- **Gate:** schema frozen; real logs flowing; baseline reproduced with numbers.

### Phase 1: Home simulator and datasets
- KNX-native simulator in layers: home generator (rooms, loads from real `loadData.json` shapes) → occupant personas and routines
  (LLM personas as priors) → device-event generator → physics only for HVAC and solar → LoadProfileGenerator for energy shapes.
- Edge-case library: guests, travel, festivals, seasons, power cuts, sensor faults, a new occupant, a habit change, multiple occupants in conflict.
- Validation report: synthetic vs real distributions; train-on-synthetic, test-on-real.
- **Gate:** 200+ diverse homes generated; validation report passes agreed thresholds.

### Phase 2: Base model pre-training (offline, PC or cloud GPU)
- Abstract event tokenizer and a compact backbone with standard ops (10–40M target).
- Heads: next action, time to next action, confidence. Calibrate with conformal prediction.
- Benchmarks: leave-one-home-out, drift splits, cold start (first 1–14 days).
- **Gate:** beats baselines on held-out homes; INT8 quantized accuracy loss within an agreed bound.

### Phase 3: On-device runtime and adaptation (hardware bake-off)
- Run the INT8 model on 2–3 candidate boards (VIM3L as the 2GB-floor test bed, plus the RK3576 and one other).
  Measure op coverage, latency and RSS.
- Adapter training on CPU: peak memory, minutes per night, forgetting with replay.
- **Gate:** fits the 2GB floor next to all OKAS services; nightly adaptation measurably improves per-home accuracy. **Outcome: input to the v2 hardware decision.**

### Phase 4: Decision and guardrail engine
- Build L0–L6 as a deterministic, separately tested module, with contract tests for every rule.
- **Shadow mode**: MSI decides but doesn't act. We compare its decisions with what users actually did.
- **Gate:** zero guardrail violations in replay and simulation; shadow precision above threshold.

### Phase 5: Pilot homes (autonomy ladder)
- Shadow → suggest (app notification) → act on low-impact, reversible loads → wider scope, per home, per load class.
- Measure acceptance, undo rate, energy deltas (A/B) and user trust.
- **Gate:** agreed undo-rate ceiling and measurable savings in pilot homes.

### Phase 6: Fleet learning and System-2
- Federated adapter aggregation (after a privacy review) to improve the base model across homes.
- Nightly System-2: habit review, drift detection, retiring stale routines.

## 5. Biggest risks
1. **On-device training tooling on aarch64**: unproven. Mitigation: spike early in Phase 3; fallback is a tiny head trained in plain C++/NumPy.
2. **Synthetic-to-real gap**: mitigated by Phase 0 real logging and validation gates.
3. **NPU operator coverage**: mitigated by choosing a standard-op backbone and testing on real boards.
4. **Trust**: one bad action (lights off on someone, AC off in an Indian summer) costs more than 100 good ones. Hence the autonomy ladder and hard comfort floors.
5. **Feedback loops**: MSI's own actions must stay out of training data.

## 6. Open questions for Iris
- Which real data can we log now, and is that OK with site owners (consent and privacy)?
- Are there occupancy, temperature or energy-meter signals on the KNX bus in typical installs? (This decides what "context" means.)
- What does MSI control: lights, curtains and HVAC only, or anything on the bus?
- How does MSI talk to the user: app notifications for suggestions and "why" explanations?
- Which market comes first (India vs elsewhere)? This sets comfort models and energy levers.

## 7. Unverified claims to check before the spec
DomusFM parameter count and latency (PDF); RK3576 dense INT8 TOPS; i.MX 95 NPU TOPS; all dataset licenses;
ExecuTorch and ONNX Runtime training on aarch64; NPU training support per vendor; energy figures marked UNVERIFIED in `streams/05`.
