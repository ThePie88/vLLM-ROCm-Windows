# 04 — Hard parts: kernel, RDNA3 ISA & quantizzazione

## RDNA3 / gfx1100 — vincoli ISA che decidono tutto

- **WMMA** (Wave Matrix Multiply-Accumulate), tile fisso **16×16×16**, **wave32**, frammenti A/B replicati sulle metà-lane. **NON c'è matrix-core dedicato**: WMMA gira sulle SIMD/VALU → throughput = quello delle istruzioni DOT, **non** il throughput MFMA di CDNA (MI300). Aspettativa di perf calibrata di conseguenza.
- **dtype WMMA**: FP16, BF16, **IU8, IU4** (accum FP16/FP32/BF16/I32). **Nessun FP8/BF8** su RDNA3 (arriva con RDNA4/gfx12 e CDNA3/4).
- **LDS = 64 KB per workgroup** (vincolo portante): qualsiasi kernel custom deve far stare i tile + stato indexer in 64 KB. **Stesso limite di CDNA3 MI325X**, dove i kernel sparse-attention DeepSeek già crashano richiedendo 96 KB → retiling obbligatorio.
- 1536 VGPR/SIMD in wave32 (fino a 12 wave/SIMD). `rocWMMA` è il wrapper C++ portabile WMMA e **supporta gfx1100/1101/1102** (ROCm 6.4+) → building block principale per kernel matrix from-scratch.

## Attention / paged-KV — il blocker (Gate C)

vLLM su RDNA3 **non** usa AITER/CK (CDNA-only): ripiega su `TRITON_ATTN` / `ROCM_ATTN` / `TORCH_SDPA`. Le rotte:

| Rotta | Pro | Contro |
|---|---|---|
| `TORCH_SDPA` (baseline) | corretto, zero kernel custom, **gira già** (SDPA verificato) | lento, non competitivo; non è paged |
| `TRITON_ATTN` (~800 LoC puro Triton, paged KV) | minor codice, vendor-portabile; Triton gira nativo da noi | FA Triton fragile su gfx1100 anche su Linux (#4514: stack-frame overflow/accuratezza); correttezza attn-scale **da provare** |
| Port HIP `__GFX11__` WMMA (`csrc/rocm/attention.cu`) | logica wave32 già scritta upstream; massima perf potenziale | mai buildato su Windows; bug DPP -O0; CMake-config HIP rotti |
| HIP WMMA **from-scratch** (seed: Repeerc FA2-RDNA3) | controllo totale; Repeerc **compila** nativo (MSVC+HIP+rocWMMA) | Repeerc **runtime via ZLUDA** (non prova launch ROCm nativo); SD-shaped (no paged KV/block-table/GQA/varlen) → va reimplementata la semantica |

**Piano**: bake-off in Fase 2 con `TORCH_SDPA` come floor; partire da `TRITON_ATTN`; in parallelo tentare il port del kernel `__GFX11__`. Validare numerica a **head_size=128 + paged KV + GQA**, non su attention dense SD-shaped. KV-write: preferire il `reshape_and_cache` **Triton** (resta C++-free, pairs con `TRITON_ATTN`).

> **Validato sul box (2026-06-29):** un GEMM **rocWMMA 16×16×16 wave32 fp16→fp32** e un kernel **RMSNorm** compilano e girano corretti su gfx1100 nativo (`experiments/phase0_kernels`, rel err 0.0 / max_err ~1e-6). Quindi la rotta **HIP-WMMA from-scratch è praticabile** sul nostro toolchain: scrivere un paged-attention WMMA a mano è un'opzione reale, non solo teorica. (Gotcha: serve `-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__` perché rocWMMA usa il ctor `__half(float)` che i flag di torch rimuovono.)

## Triton su Windows nativo

- Esiste un path **HIP nativo (non ZLUDA)**: `woct0rdho/triton-windows` PR #179 (merged 2025-12-30), usa wheel TheRock + clang-cl del HIP SDK + `rocm_sdk.find_libraries()`. **Il nostro `triton 3.6` compila nativo** (provato).
- Caveat: la produzione del backend AMD-Windows è "agli inizi" upstream; correttezza/perf delle **kernel attention/MoE** su gfx1100 è il rischio reale (oltre il vector-add che abbiamo validato). Va testato kernel-per-kernel.
- Path ZLUDA (`lshqqytiger`, `Repeerc`) = **solo fallback** di bring-up; contraddice l'obiettivo "ROCm nativo" → non per la v1.

## Quantizzazione su RDNA3 (budget 20 GB)

| Metodo | Stato su gfx1100 | Decisione |
|---|---|---|
| **AWQ / GPTQ (Triton)** | `VLLM_USE_TRITON_AWQ` auto su ROCm; via 4-bit principale; `gptq_gemm` flagged buggy, MoE Triton SiLU-only | **path primario**; preferire **AWQ dense** prima; validare numerica su gfx1100 |
| INT4/INT8 **WMMA** (W4A16/W8A8) | IU4/IU8 supportati; **INT8 WMMA (iu8→i32) validato su gfx1100 nativo: max_abs_diff 0** (`experiments/phase0_kernels`); kernel HIP recenti (#41394 W4A16, #44075 MoE W4A16) | **target nativo confermato** — il building block intero funziona; resta da compilare i kernel vLLM reali |
| **Marlin** / AWQ-/GPTQ-Marlin | CUDA-only (SM≥8.0), nessun ROCm | **stub** → fallback Triton |
| **FP8 (W8A8)** | hardware assente su RDNA3 | **stub** (`supports_fp8()` già False su gfx11); usa fp16/bf16 |
| **MXFP4** | hardware assente (solo CDNA4+) | **non disponibile**; modelli MXFP4 falliscono/dequantizzano |
| **bitsandbytes** | upstream Linux-only; fork `0xDELUXA/bitsandbytes_win_rocm` (gfx1100-1103) | **opzionale**, no prova vLLM-compat → preferire AWQ/GPTQ |

## Distributed (single-GPU = tutto stubbabile)

- vLLM **corto-circuita ogni collettiva a `world_size==1`** (PyNcclCommunicator e CustomAllreduce si auto-disabilitano). Custom all-reduce HIP è gated a CDNA (gfx94x/95x), mai RDNA3.
- **RCCL assente su Windows**; **`torch.distributed` è CONFERMATO ASSENTE** nel nostro wheel (`is_available()==False`, ROCm #5689) → `init_distributed_environment` (che chiama `init_process_group` anche a world_size==1) **solleverà prima** di qualsiasi corto-circuito interno. Mitigazione **obbligatoria**: **hard short-circuit incondizionato** dell'init quando `world_size==1` su Windows (nel platform plugin / patch).
- **TP/PP = fuori scope v1** (enforce `tensor_parallel_size=1`).

## hipBLASLt gap
Report indicano che l'immagine vLLM ROCm "non supporta hipBLASLt per gfx1100 e ripiega su hipBLAS" → impatto perf GEMM. Validare la copertura tunata fp16/bf16 su gfx1100 prima di assumerla.
