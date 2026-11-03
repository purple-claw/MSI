# Stream 2: On-device continual learning (raw research report, 2026-11-03; search quota ran out after ~10 queries; UNVERIFIED items not checked)

## On-device training
- MCUNetV3 "On-Device Training Under 256KB Memory" (NeurIPS 2022, https://arxiv.org/abs/2206.15472): quantization-aware scaling + Sparse Update. CNN/vision only; RNN/Transformer = future work. Code https://github.com/mit-han-lab/tiny-training (UNVERIFIED live).
- PockEngine (MICRO 2023, https://arxiv.org/abs/2310.17752): compile-time sparse backprop; targets Jetson/MCU, not NPU boards. Check figures in paper.
- TinyTL (NeurIPS 2020, https://arxiv.org/pdf/2007.11622): ACTIVATIONS dominate training memory, not params. Freeze weights, train biases + lite residual. Key lesson for our adapters.
- LoRA (Hu 2021) / BitFit (Ben Zaken 2021): standard refs, UNVERIFIED in this pass.
- ExecuTorch training: docs 404; ZeroLock (https://arxiv.org/pdf/2608.07974) ran LoRA backward + AdamW on-device with custom work. Possible, not turnkey. ONNX Runtime training, TFLite on-device training, MNN, ncnn: NOT verified.
- NPU training support: not confirmed; assume training on CPU.

## Continual learning
- EWC (https://arxiv.org/abs/1612.00796); 2025 replication (https://arxiv.org/abs/2507.10485): reduces forgetting, slight plasticity cost; Fisher memory -> adapters only.
- Replay reservoir + adapter-per-period: standard, UNVERIFIED edge numbers.

## Federated (later)
- Flower LoRA FedAvg example (https://flower.ai/apps/aashmohd/diffusion_example).
- FLoCoRA (https://arxiv.org/abs/2310.13283): exchange only quantized LoRA adapters (2/4/8-bit).
- FLoRIST (https://arxiv.org/abs/2506.09199): SVD aggregation beats naive LoRA averaging.

## Sequence models
- Mamba INT8: eMamba (https://arxiv.org/pdf/2508.10370); UW capstone: Mamba-2 on Orange Pi 5 Ultra, NPU lacked custom ops (not peer-reviewed); TU Delft thesis Mamba on TFLM 60.4KB peak RAM. Risk: scan ops won't map to NPU.
- TCN/GRU/tiny transformer: not searched; expectation (UNVERIFIED) standard ops map better to NPU INT8.

## Distillation
- Surveys: https://arxiv.org/pdf/2503.12067, https://arxiv.org/pdf/2505.02309, https://arxiv.org/pdf/2410.11845.

## Recommendations (subagent)
1. Frozen base, train small adapters (LoRA or bias-only) on CPU; budget ACTIVATION memory (batch 1, short windows, checkpointing); measure peak RSS.
2. Prefer TCN/GRU/standard-op backbone; run NPU op-coverage test on the real board before committing.
3. Forgetting: replay reservoir + adapter snapshots; EWC on adapters only if needed.
4. Base-model data: teacher LLM + simulated homes, validated on held-out sims + small real set.
5. Defer federated; when added, quantized adapters only + privacy review.

## Open items
ExecuTorch / ORT on-device training on aarch64; NPU op list; primary-paper numbers; LoRA/BitFit refs.
