# Stream 5: Energy optimisation and simulators (raw research report, 2026-11-03; abstracts/snippets; UNVERIFIED items not checked)

## Simulators / generators
- OpenSHS (https://www.mdpi.com/1424-8220/17/5/1003): 3D ADL event simulator; license UNVERIFIED.
- SIMADL (https://www.mdpi.com/2306-5729/3/2/11): OpenSHS-generated, CC BY-SA 4.0.
- VirtualHome (MIT): task-level activities, not device traffic; license check needed.
- Sinergym (https://github.com/ugr-sail/sinergym): Gymnasium over EnergyPlus, MIT (UNVERIFIED).
- BOPTEST (https://github.com/ibpsa/project1-boptest): building-control benchmark.
- CityLearn v3 (https://arxiv.org/pdf/2609.21570): district scale.
- obFMU/obXML (LBNL): occupant behaviour co-sim with EnergyPlus.
- LoadProfileGenerator (https://github.com/FZJ-IEK3-VSA/LoadProfileGenerator): MIT CONFIRMED; agent-based residential profiles. Best fit for load layer.
- LLM household sims: SmartLLM (IEEE IoT J), "Simulating the Resident" (https://arxiv.org/abs/2607.08231), S5-HES (https://arxiv.org/abs/2603.01554, mapping UNVERIFIED). Prototypes.
- No KNX-specific simulator found.

## Validation
- Sim-to-real HAR: VirtualSmartHome digital twin (Sensors 2023); ACM DOI 10.1145/3665331; SynSys HMM (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6427177/); Poisson calibration (Remote Sensing 2020). Balanced accuracy on imbalanced real test.
- Checks: per-device on-time histograms, inter-event gaps, daily energy, real-vs-synthetic discriminator AUC.

## Energy savings evidence (study-specific)
- Lighting occupancy: ~24% (LBNL meta-analysis, URL UNVERIFIED); private offices 20-26% (https://eta-publications.lbl.gov/publications/comparison-control-options-private); off-delay setting matters a lot.
- Occupancy HVAC: 5-21% residential, wider in commercial; baseline matters.
- Learning thermostats independent: ~5% (BPA Nest pilot https://www.bpa.gov/-/media/Aep/energy-efficiency/emerging-technologies/ET-Documents/BPA-FPUD-Nest-Thermostat-Pilot-Savings-Assessment.pdf). Vendor claims not independent.
- Shades: FLEXLAB cooling 19-38% + lighting 49-76% combined (https://flexlab.lbl.gov/publications/blind-blinds-opening-our-eyes-savings); a sunny-climate field study found no solar-gain benefit vs manual.
- Standby: 4% (Turkey, 260 homes) to 5-26% (CA, 10 homes).
- ToU shifting: appliance-only up to ~6%; with storage much more.
- India BEE: ~6% per degree C AC setpoint; default 24 C (secondary source).
- Measuring: rated W x on-time is weak (1-32% error); prefer whole-home meter A/B, sim counterfactual (same occupant trace on/off), label estimates with error bands.

## Comfort bounds
- ASHRAE 55 adaptive 80%: upper 0.31*Tout_mean + 21.3 C, lower +14.3 C (2010 addendum; verify vs 55-2023). pythermalcomfort library.
- IMAC India (https://opus.cloud1.lib.uts.edu.au/handle/10453/41878): neutral 21.5-28.7 C for outdoor 13-38.5 C (AC/mixed); IMAC Assistant tool.

## Recommendations (subagent)
1. Layered simulator: KNX-native device-event generator + occupant schedule layer (LLM personas as priors, validated) + physics only for HVAC/solar (Sinergym or reduced-order) + LoadProfileGenerator.
2. Distribution-match report + train-synthetic/test-real before training.
3. Tag synthetic vs real data separately.
4. Report savings as paired counterfactual deltas (sim) and A/B or diff-in-diff (field).
5. Hard comfort floors (IMAC India, ASHRAE 80% elsewhere). Start with best-evidence levers: lighting occupancy/off-delay, HVAC setback when unoccupied, standby/forgotten-load alerts.
