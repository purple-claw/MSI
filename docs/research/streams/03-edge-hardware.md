# Stream 3: Edge NPU boards and runtimes (raw research report, 2026-11-03; snippet-level; prices/power NOT verified)

## SoC table (INT8 TOPS etc.; UNVERIFIED where not sourced)
| Part | NPU TOPS | SDK | Mainline NPU | Notes |
|---|---|---|---|---|
| RK3566 | 0.5 or 0.8 (conflicting) | RKNN-Toolkit2 | ? | |
| RK3568 | ? | RKNN-Toolkit2 | NPU driver RFC under review (https://lwn.net/Articles/1076412/) | |
| RK3576 | 6 (sparsity-asterisked; INT4/8/16, FP16, BF16) (https://armdesigner.com/download/RK3576_Brief_Datasheet_V1.2-20240311.pdf) | RKNN-Toolkit2 2.3.x, RKNN-LLM | Rocket probes on linux-next, inference untested | Flash Attention + W4A16 support in toolkit |
| RK3588S | 6 (A76+A55) | RKNN-Toolkit2, RKNN-LLM | Rocket reported merged 6.18 + Mesa 25.3 (Phoronix, secondary) | more RAM/cost than needed |
| i.MX 8M Plus | 2.3 (module vendor claim) | eIQ | ? | NXP longevity 10-15y |
| i.MX 93 | 0.5 Ethos-U65 microNPU | Vela/eIQ | not Linux-visible accel | weak |
| i.MX 95 | 2 (2023 press) eIQ Neutron | eIQ | ? | NXP longevity |
| A311D | 5 (VeriSilicon) | Amlogic SDK; Teflon/Etnaviv in Mesa (slower) | yes via Teflon | Khadas VIM3 ecosystem |
| S905D3 | ? | ? | ? | current v1.2 board |
| Genio 350/510/700 | ?/3.2/4.0 | NeuroPilot; LiteRT + ExecuTorch backends | ? | Yocto from MediaTek |

## Operators
- RKNN-Toolkit2 release notes: LayerNorm, LSTM, Transpose, MatMul optimisations (2.3.0), SDPA (2.0.0b0), Flash Attention RK3562/RK3576 only (2.1.0). Per-op INT8 table not found.
- Ethos-U65: LSTM/GRU/transformer ops UNVERIFIED; use `vela --supported-ops-report`.
- NeuroPilot / Teflon op coverage UNVERIFIED.

## Runtimes
- ONNX Runtime: Rockchip NPU EP (preview); no NXP/Amlogic/MediaTek EP seen.
- LiteRT: NPU vendors Google, Qualcomm, MediaTek, Intel.
- ExecuTorch 1.0: Ethos-U (production), NXP, MediaTek, Qualcomm; CPU fallback via XNNPACK. Broadest embedded NPU list.
- TVM: Ethos-U BYOC (U55 first).

## Shortlist (subagent)
1. RK3576 (6 TOPS sparse, best transformer toolkit, mainline driver emerging; dense TOPS + price UNVERIFIED)
2. NXP i.MX 95 (longevity; weak/unverified NPU)
3. MediaTek Genio 510 (3.2 TOPS, LiteRT + ExecuTorch paths, Yocto; closed SDK)
Runner-up RK3588S; A311D closest to current Khadas line.

## Not covered
Power, prices, RK3566/68 specs, S905D3 NPU, Amlogic longevity. Need datasheets + distributor quotes before deciding.
Key sources: https://github.com/airockchip/rknn-toolkit2/releases, https://docs.pytorch.org/executorch/1.0/backends-overview.html, https://onnxruntime.ai/docs/execution-providers/, https://developers.google.com/edge/litert/next/npu, https://nxp.com/products/product-information/product-longevity, https://www.phoronix.com/news/Rocket-Rockchip-NPU-Driver
