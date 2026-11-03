# Stream 1: Smart-home behaviour learning (raw research report, 2026-11-03; claims marked UNVERIFIED not checked)

## Datasets
- CASAS (WSU): ambient PIR/door/env sensors, single+multi-resident; Aruba, Cairo, Kyoto7/8, Milan (https://arxiv.org/pdf/2311.09514). Sandia page cites longitudinal release of 189 homes 2007-2024 (https://www.sandia.gov/raiice/?p=6351, UNVERIFIED). License UNVERIFIED.
- ARAS: 2 real houses, multi-resident, ~2 months (https://user.ceng.metu.edu.tr/~alemdar/publication/2013_c_pervasivehealth/). License UNVERIFIED.
- Orange4Home: single occupant, 236 sensors, 4 weeks (~180h). URL UNVERIFIED.
- MARBLE: smartwatch+env sensors, ~16h, 12 subjects, 13 ADLs, simulated home (https://everywarelab.di.unimi.it/index.php/research/datasets/220-marble-dataset-multi-inhabitant-activities-of-daily-living-combining-wearable-and-environmental-sensors-data).
- ContextAct@A4H: ADL + context vars (weather, temp, noise, humidity, visitors) (https://data.mendeley.com/datasets/fcj2hmz5kb).
- MuRAL (2025-26): multi-resident, 23 sensors, NL annotations (https://mural.imag.fr, not fetched).
- No public SmartThings routine-log dataset found. No KNX-specific public dataset found.

## Habit mining / next-action
- SmartSense (https://arxiv.org/pdf/2208.06089): self-attention over (device, control, dow, hour) + commonsense from routines; size UNVERIFIED.
- DeepUDI (AAMAS 2023, https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1634.pdf) / SmartUDI (DOI 10.1145/3610906): routine+intent+multi-level periodicity; SmartUDI claims edge-deployable. Code/size UNVERIFIED. Best-matched to our problem.
- Timing Matters (https://arxiv.org/abs/2411.18719): transformer predicting time of next action.
- DomusFM (https://arxiv.org/abs/2602.01910, 2026): event-based foundation model for smart homes, leave-one-dataset-out on 7 datasets, claims edge-deployable. Size UNVERIFIED. Top candidate to study.
- LLM agents (SAGE https://arxiv.org/abs/2311.00772, Sasha UNVERIFIED): cloud-scale, not edge-viable.

## Drift
- Routine Computing review (https://arxiv.org/pdf/2604.21934, 2026): best entry point.
- HAR survey (https://arxiv.org/pdf/2111.04418); Azefack 2019 (https://hal-emse.ccsd.cnrs.fr/emse-03128531); DISCOVER (https://arxiv.org/pdf/2503.01733v3); Gama drift overview (https://arxiv.org/pdf/1010.4784).
- Gap: no benchmark of drift adaptation on public smart-home data -> build our own.

## Cold start
- READY / user-guided personalization, Ali/Augusto/Windridge, UAIS 2023, DOI 10.1007/s10209-022-00883-x.
- Home-to-home transfer, Rashidi & Cook (https://cdn.aaai.org/ocs/ws/ws0562/1995-8504-1-PB.pdf).

## Recommendations (subagent)
1. Small attention next-event model (SmartSense/SmartUDI-style, time embeddings) as baseline; pretrain CASAS/ARAS, leave-one-home-out.
2. Add time-to-next-action head; evaluate DomusFM.
3. No LLM on the on-device path.
4. Build our own drift benchmark; per-home adapters / sliding windows.
5. Cold start: population prior + confidence-gated automation + user approval. Collect KNX data early.
