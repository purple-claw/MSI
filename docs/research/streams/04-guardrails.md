# Stream 4: Guardrails and safe autonomy (raw research report, 2026-11-03; 15 searches, snippet-level, no full-text reads)

## Safety architectures
- Shielding, Alshiekh et al. AAAI 2018 (https://arxiv.org/abs/1708.08611): temporal-logic shield; preemptive (safe action set) or post-posed (substitute safe action).
- IoTGuard NDSS 2019 (https://www.ndss-symposium.org/ndss-paper/iotguard-dynamic-enforcement-of-security-and-safety-policy-in-commodity-iot): runtime policy enforcement, 11 policies, ~17% overhead.
- IoTSafe NDSS 2021 (https://ndss-symposium.org/ndss-paper/iotsafe-enforcing-safety-and-security-policy-with-real-iot-physical-interaction-discovery): discovers physical interactions, predicts risky states.
- Soteria / IoTSan: multi-app model checking (primary URLs not retrieved).
- CBF safety filters: UNVERIFIED URLs.
- HVAC safe RL: safety layer (https://www.arxiv.org/pdf/2409.08132); RL breaks comfort during exploration (Energies 2025); live chiller with pessimistic margin + fallback, 9-13% savings (via survey https://arxiv.org/pdf/2603.08956); batch learning 16.7% (ICCPS 2022).
- Whitelists, rate limits, quiet hours, obstacle checks: no source, engineering practice.

## Confidence-gated autonomy
- Parasuraman/Sheridan/Wickens 2000 levels of automation (https://doi.org/10.1109/3468.844354).
- Conformal prediction as abstention (https://arxiv.org/pdf/2107.07511 tutorial; selective CRC https://arxiv.org/pdf/2508.07556).

## Implicit feedback
- Undo/override as signal: only patents found (no efficacy evidence).
- Exposure bias: unseen action != rejected (Ermis et al. https://arxiv.org/pdf/2004.13106).
- Bandits for energy: Bregere ICML 2019 (https://proceedings.mlr.press/v97/bregere19a.html). No LinUCB-from-overrides smart-home paper found. OPE/IPS: gap.

## Trust / UX
- IMWUT 2025 "House That Saves Me" (https://www.peasec.de/paper/2025/2025_HenkelHaeslerAlNajmiHesselReuter_HouseThatSavesMe_IMWUT.pdf): users prefer more automation in warnings but want customization.
- Multi-occupant conflicts are top user concern (questionnaire, UNVERIFIED URL).

## Failure modes
- Hidden feedback loops, Sculley NeurIPS 2015.
- Multi-user conflict frameworks (https://peerj.com/articles/cs-1443).
- Privacy: timing leaks occupancy (Peek-a-Boo https://arxiv.org/pdf/1808.02741; PFirewall https://arxiv.org/pdf/2101.10522).
- Sensor faults, KNX scene conflicts: gaps.

## Proposed layered guardrails (subagent)
- L0 hard envelope: deterministic limits (setpoint min/max, curtain travel, stall), non-learned.
- L1 policy shield: temporal-logic/rule specs (heating when away, window open + HVAC), preemptive.
- L2 rate limits, quiet hours, whitelist, per-device cooldown, action budget.
- L3 confidence gate: conformal/selective threshold -> suggest / ask / act-with-reason / act silently (only low-impact, reversible, repeatedly accepted).
- L4 conflict/occupant arbitration: never override a manual change within a hold period.
- L5 feedback with exposure: only exposed actions count as negatives; quarantine own actions from training data; kill switch + fallback.
- L6 sensor-fault rejection + "why" record per action; local logs, minimal retention.
- Eval: offline replay, envelope contract tests, shadow mode before acting.
