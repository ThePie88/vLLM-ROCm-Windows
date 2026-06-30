# 02 — Architettura vLLM & matrice di porting

## vLLM main è V1-only e quasi tutto Python

`main` (giugno 2026) ha rimosso il motore V0 (RFC #18571). Lo stack engine vive sotto `vllm/v1/` ed è **puro Python, hardware-agnostic**:

- `vllm/v1/engine/` — `llm_engine.py`, `async_llm.py`, `core.py` (EngineCore), `core_client.py`
- `vllm/v1/core/` — scheduler (`sched/`), `kv_cache_manager.py`, `kv_cache_coordinator.py`, `block_pool.py`, `kv_cache_interface.py`
- `vllm/v1/executor/` — `uniproc_executor.py`, `multiproc_executor.py`, `ray_executor.py`
- `vllm/v1/worker/` — `gpu_worker.py`, `gpu_model_runner.py` (condivisi CUDA/ROCm)

**ROCm non è un device-type separato**: `RocmPlatform` imposta `device_type="cuda"`, `dispatch_key="CUDA"` → riusa lo **stesso** `gpu_worker.py`/`gpu_model_runner.py` di NVIDIA via `torch.cuda` (HIP-as-CUDA). Non esiste `rocm_worker.py`. **Conseguenza enorme: il layer Python porta ~1:1.**

## I 3 seam dove si attacca un backend Windows-ROCm

1. **Platform** (`vllm/platforms/interface.py:Platform`) — unica superficie hardware. Detection via entry-point group **`vllm.platform_plugins`**: un plugin out-of-tree **sovrascrive** i built-in senza forkare il core. Un backend implementa ~10-15 classmethod.
2. **Attention backend registry** (`vllm/v1/attention/backends/registry.py`) — `Platform.get_attn_backend_cls()` ritorna un **path-stringa** risolto da `resolve_obj_by_qualname`; `register_backend()` permette override a runtime (`AttentionBackendEnum`: `ROCM_ATTN`, `TRITON_ATTN`, `ROCM_AITER_FA`, `CUSTOM=None`, …).
3. **Custom-ops dispatch** (`vllm/_custom_ops.py`) — chiama `torch.ops._C / _rocm_C / _moe_C / _C_stable_libtorch`, popolati dai moduli nativi HIP/C++. Ogni op è guardata da `hasattr(...)` → **stubabile**.

## Strategia di porting: plugin out-of-tree, non fork

Il piano cardine è **non forkare vLLM core**. Si crea un package esterno che registra un `vllm.platform_plugins` entry-point con una sottoclasse di `RocmPlatform` che:
- rileva via `torch.cuda.is_available()` + `torch.cuda.get_device_properties().gcnArchName` **invece di `amdsmi`** (assente su Windows);
- corto-circuita l'init del process-group quando `world_size==1`;
- forza `use_custom_allreduce()->False`, disabilita AITER/FP8 su gfx11;
- registra il backend di attention scelto.

Le uniche modifiche "invasive" inevitabili sono al **build system** (`setup.py`/`CMakeLists.txt`), che vanno patchati o bypassati.

## Matrice di porting

Status: `1:1` portabile · `adatta` · `da-zero` · `stub` (disabilita). La colonna "target" è il comportamento di riferimento Linux+NVIDIA/CDNA.

| Componente | Impl upstream | Stato su Windows-ROCm-gfx1100 | Strategia |
|---|---|---|---|
| LLMEngine / AsyncLLM / EngineCore | Python (`vllm/v1/engine`) | Portabile; attrito solo su multiprocessing spawn-vs-fork | **1:1**, preferisci uniproc/multiproc a Ray |
| Scheduler / KV-cache manager / block pool | Python (`vllm/v1/core`) | Nessun codice GPU | **1:1** |
| Executor (uniproc/multiproc) | Python, spawn + ZMQ | spawn funziona su Windows; Ray opzionale | **1:1** |
| Platform layer | `interface.py` + `rocm.py` | `rocm.py` usa `amdsmi` (assente su Windows) | **adatta**: platform plugin out-of-tree, detection via `torch.cuda` |
| GPU worker + model runner | condivisi (CUDA path) | Portabili; CUDA-graph capture può rompersi su HIP-Windows | **adatta**: invariati, ma `enforce_eager`, niente cudagraph all'inizio |
| **Attention (RDNA3)** | `__GFX11__` WMMA in `csrc/rocm/attention.cu`; `TRITON_ATTN` (~800 LoC); `ROCM_ATTN` (HIP+Triton ibrido) | C++ buildato solo su Linux; Triton-attn fragile su gfx1100 anche Linux (#4514) | **da-zero/adatta (GATE C)**: `TORCH_SDPA` baseline → `TRITON_ATTN` → port HIP `__GFX11__` |
| `reshape_and_cache` (KV write) | forma HIP C++ + forma Triton | stesso rischio build dell'attention | **adatta**: preferisci il writer Triton (C++-free), pairs con `TRITON_ATTN` |
| GEMM / BLAS | rocBLAS / hipBLAS / hipBLASLt | Su Windows presenti; copertura hipBLASLt gfx1100 incerta (fallback hipBLAS) | **1:1/adatta**: consuma dai wheel; valida hipBLASLt fp16/bf16 |
| Quant — Marlin / AWQ-Marlin / GPTQ-Marlin | `.cu` CUDA-only (SM≥8.0) | Nessun Marlin ROCm | **stub**; fallback Triton AWQ/GPTQ |
| Quant — AWQ / GPTQ (Triton) | `VLLM_USE_TRITON_AWQ` auto su ROCm | Dipende da Triton-ROCm-Windows (codegen base , attn-scale TBD); `gptq_gemm` flagged buggy, MoE SiLU-only | **adatta**: path 4-bit primario; preferisci AWQ dense prima |
| Quant — FP8 (W8A8) | `supports_fp8()` solo gfx9/gfx12 | RDNA3 **non ha** matrix FP8 | **stub**: usa fp16/bf16 + INT4/INT8 WMMA |
| Quant — bitsandbytes | backend ROCm Linux-only | fork community `0xDELUXA/bitsandbytes_win_rocm` (gfx1100-1103) ma no prova vLLM-compat | **opzionale/stub**: preferisci AWQ/GPTQ |
| AITER / CK-FA / QuickReduce | CDNA (gfx9) only | hard-gated gfx9; irrilevante RDNA3 | **stub** (`VLLM_ROCM_USE_AITER=0` default) |
| Distributed — RCCL / pynccl / custom-AR | `dist_backend="nccl"` (RCCL) | RCCL **assente su Windows**; `torch.distributed` forse assente (ROCm #5689) | **stub v1**: `world_size==1` corto-circuita; hard short-circuit init; TP/PP fuori scope |
| Build — `setup.py` | forza `VLLM_TARGET_DEVICE="empty"` su win32 | hard blocker | **adatta**: patch per win32 + `torch.version.hip`; inietta clang++ HIP SDK |
| Build — `CMakeLists`/hipify | Unix-only (`/opt/rocm`, GCC, hipify POSIX) | zero handling WIN32/MSVC; CMake-config HIP SDK Windows rotti | **da-zero (harness Windows)**: patch CMake-config OPPURE bypassa con `torch.utils.cpp_extension` per un subset minimo. `cmake/hipify.py` (Python) è **1:1 portabile** |

## Riferimenti di codice utili
- Kernel RDNA3 già in-tree: `csrc/rocm/attention.cu` (path `__GFX11__` wave32 WMMA), `csrc/rocm/q_gemm_rdna3.cu`, `q_gemm_rdna3_wmma.cu`, `moe_q_gemm_rdna3.cu`, `skinny_gemms.cu`. gfx1100-1103 in `HIP_SUPPORTED_ARCHS` (gfx1102/1103 aggiunti PR #40037, apr 2026).
- Precedente di port: **`SystemPanic/vllm-windows`** builda vLLM nativo su Windows **per CUDA/NVIDIA** — non ci aiuta su ROCm, ma **prova che la csrc di vLLM è portabile a Windows**. Riferimento prezioso per le patch di build.
- **`lemonade-sdk/vllm-rocm`**: NON è un fork sorgente, è un **repackager Linux-only** (scarica wheel AMD prebuilt). Utile solo per metodologia/qualification, zero codice Windows.
