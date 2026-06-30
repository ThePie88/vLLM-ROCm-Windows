# vLLM su Windows nativo + AMD ROCm (RDNA3 / gfx1100) — Fondazione di ricerca

> Obiettivo: far girare **vLLM su Windows nativo** (niente WSL2) con **AMD ROCm** su **RDNA3** (RX 7900 XT = `gfx1100`), portando 1:1 dove possibile e **costruendo da zero** ciò che non può funzionare. Target: massima precisione/velocità, parità col path CUDA di riferimento. Per ora **single-GPU** (multi-GPU = fuori scope, vedi sotto).

Questi documenti sono la sintesi di una ricerca multi-agente (10 dimensioni, verifica avversariale) **fusa con i risultati misurati sul ferro reale** del progetto. Dove web e misura divergono, **vince la misura**.

## TL;DR — la fattibilità è confermata, il baricentro è l'attention

1. **Le fondamenta esistono e funzionano sul nostro hardware** (misurato 2026-06-29): `torch 2.10.0+rocm7.13` con `torch.cuda.is_available()==True` su gfx1100, matmul fp16 OK, `SDPA` OK; **Triton 3.6 compila ed esegue kernel nativi** (no ZLUDA); toolchain HIP nativa completa (HIP SDK 7.2 / AMD clang 22 + MSVC 14.44 + Win SDK 10.0.26100).
2. **vLLM è quasi tutto Python e platform-agnostico** (engine V1, scheduler, KV-cache manager, executor). ROCm è già una piattaforma "CUDA-alike" → riusa lo stesso `gpu_worker.py`/`gpu_model_runner.py` di NVIDIA. Il porting del layer Python è ~1:1 attraverso **un solo seam**: la classe `Platform`.
3. **Il vero lavoro** è: (a) il build delle estensioni native HIP su Windows, (b) un **path di attention/paged-KV** corretto e veloce su gfx1100, (c) un **platform plugin** out-of-tree che bypassa `amdsmi` e le assunzioni Linux.
4. **RDNA3 non ha matrix-core né FP8** (usa WMMA sulle SIMD): niente AITER/CK/MFMA, niente Marlin (CUDA-only), niente FP8. La via quantizzazione è **AWQ/GPTQ via Triton + INT4/INT8 WMMA**, dtype principali fp16/bf16.

## Indice

| Doc | Contenuto |
|---|---|
| [01 — Feasibility & verdetto](01-feasibility-and-verdict.md) | Verdetto, ground-truth vs ricerca, critical path, il blocker n.1 |
| [02 — Architettura & matrice di porting](02-architecture-and-port-matrix.md) | I 3 seam di vLLM, strategia platform-plugin, matrice componente-per-componente |
| [03 — Ambiente, toolchain & build](03-environment-toolchain-build.md) | Stack verificato, toolchain HIP+MSVC, harness di build, trappole Windows |
| [04 — Hard parts: kernel & quant](04-hard-parts-kernels-quant.md) | Attention WMMA/Triton, RDNA3 ISA, quantizzazione, distributed |
| [05 — Roadmap, gate & rischi](05-roadmap-gates-risks.md) | Fasi con go/no-go, rischi ordinati, domande aperte |
| [06 — Enhancement SOTA](06-enhancements-sota.md) | DeepSeek attention (MLA/NSA/DSA), TurboQuant, il "metodo cinese", backlog P0/P1/P2, stack quant |

## Fonti
Ricerca grezza (digest completo): vedi gli output workflow in `../` (scratchpad di sessione). Ogni claim forte nei doc è tracciabile alle fonti citate nella ricerca (AMD docs, vLLM source/issues, TheRock, ecc.).
