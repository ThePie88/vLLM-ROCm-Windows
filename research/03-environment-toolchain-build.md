# 03 — Ambiente, toolchain & build

## Stack verificato sul ferro (2026-06-29)

| Pezzo | Versione / path | Note |
|---|---|---|
| GPU | **AMD Radeon RX 7900 XT = `gfx1100`** | RDNA3, wave32, 42 CU, **64 KB LDS/workgroup**, ~20 GB VRAM |
| PyTorch | **`torch 2.10.0+rocm7.13.0a20260508`** (HIP rt `7.13.26176`) | `cuda.is_available()=True`, fp16 matmul OK, SDPA OK. torchvision/torchaudio rocm idem |
| Triton | **`triton-windows 3.6.0.post26`** | compila+esegue kernel nativo su gfx1100 (no ZLUDA) |
| HIP SDK | **7.2.0** in `C:\HIP-SDK` | `hipcc` = **AMD clang 22.0.0**, target `x86_64-pc-windows-msvc`; `hipify-clang`, `hipInfo`, `hipconfig` presenti |
| MSVC | **14.44.35207** in `E:\BuildTools` | `cl.exe` Hostx64\x64 ok |
| Windows SDK | **10.0.26100** | `C:\Program Files (x86)\Windows Kits\10` |
| CMake/Ninja | da WinLibs/mingw64 su PATH | Git presente |
| Python | **3.12.7** | sistema, **senza venv** |

> Stack **TheRock-class `rocm7.13`** (nightly/preview), più nuovo della "PyTorch on Windows Edition 7.2.1" ufficiale AMD. Più capace, ma da **pinnare**: driver Adrenalin + ROCm/HIP + torch + triton vanno bloccati a una combinazione nota-buona (i rename DLL tra 7.1.1↔7.2 rompono i consumer). `vllm` **non** installato (lo costruiamo).

## Due stack ROCm-on-Windows (non confonderli)

1. **HIP SDK ufficiale** (installer, top 7.1.1): HIP runtime closed (`amdhip64_*.dll`) + math/primitive libs (rocBLAS, hipBLAS, hipBLASLt, rocSPARSE, rocFFT, rocSOLVER, rocRAND, rocPRIM/hipCUB, rocThrust, rocWMMA). **NON** include MIOpen/MIGraphX/RCCL/PyTorch. Supporta gfx1100/1101/1102 a tier HIP SDK; **gfx1103 no**.
2. **TheRock** (build system open AMD): builda molto di più nativo su Windows con MSVC + amd-llvm (clang/lld-link), **incluso** MIOpen, CK, rocWMMA, e i **wheel PyTorch**. È lo stack su cui siamo (rocm7.13). RDNA3 dGPU = "Build Passing" ma non "Sanity-Tested" ufficialmente → **noi siamo la sanity-test**.

**RCCL resta assente in entrambi** → multi-GPU fuori scope su Windows.

## Toolchain di build HIP nativa

- **`torch.utils.cpp_extension` ha già il path HIP-Windows nativo** (PR #150180, ~apr 2025): seleziona `hipcc.exe` per HIP SDK ≥6.4, applica `win_hip_flags`, linka `amdhip64/c10_hip/torch_hip`, emette `.pyd` con `/DLL`, imposta `-fms-runtime-lib` (ABI MSVC). **È nel nostro torch 2.10.** → è la **via a minor rischio** per buildare le custom ops.
- Su Windows il device code è clang/amdclang++ → lld; l'host `.pyd` è MSVC-ABI. **CMake vieta di mischiare command-line stile-Clang e stile-MSVC** → usare clang++ come unico compilatore per CXX e HIP.
- `cmake/hipify.py` di vLLM è Python (non perl) → **portabile 1:1** (solo normalizzare i separatori path).

## Piano harness di build (raccomandato)

**Non riusare `setup.py`/`CMakeLists` di vLLM as-is.** Strategia incrementale:
1. **PoC**: buildare+importare una `.pyd` HIP **multi-file** via `torch.utils.cpp_extension` (Gate A). *Esperimento singolo più importante del progetto.*
2. Compilare **solo le op che servono** (paged attention + un paio di norm), instradando il resto via Triton/PyTorch.
3. **Sempre Release/-O2+, mai -O0** (bug DPP gfx1100 a -O0, chiuso "not planned" → permanente).
4. Patchare `setup.py` per accettare `win32` + `torch.version.hip` (e `VLLM_TARGET_DEVICE=rocm`); gate-out RCCL/Marlin/custom-AR nel build.

## Trappole Windows note (mettere in checklist)
- **Long Path** registry + **Developer Mode** (symlink) attivi; build dir vicino alla radice del drive; **niente spazi** nei path ROCm/repo.
- `ccache/sccache` **off** per rocBLAS/rocSPARSE (errori duplicate-symbol lld-link); compiler-launcher opzionale.
- CMake-config del HIP SDK Windows storicamente **rotti/non-relocatable** (`hip-lang-config.cmake`, AMDDeviceLibs, amd_comgr) → o si patchano (stile StreamHPC) o si evita `enable_language(HIP)` compilando i `.hip` con custom command su `amdclang++`, o si usa `cpp_extension`.
- `rocWMMA`: Repeerc dovette copiare a mano gli header nel HIP SDK include dir — verificare che il nostro 7.2 li esponga per gfx1100.
- `hipBLASLt`: copertura kernel tunati per **gfx1100** incerta (report di fallback a `hipBLAS`); validare fp16/bf16 prima di farci affidamento per la perf.

## Gate A — SUPERATO (2026-06-29): ricetta funzionante per `.pyd` HIP

Costruita+importata una estensione HIP **multi-file** via `torch.utils.cpp_extension.load`, eseguita su gfx1100 (`max_err 0.0`, build ~102s). PoC nel repo: `experiments/phase0_hip_ext/`. Ricetta:
1. **Ambiente MSVC**: eseguire dentro `cmd /c` dopo `"E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"` (il warning `vswhere.exe not recognized` è innocuo, l'env x64 si inizializza).
2. **Env**: `ROCM_HOME=ROCM_PATH=HIP_PATH=C:\HIP-SDK` (così `cpp_extension` trova `C:\HIP-SDK\bin\hipcc.exe`).
3. **Bug hipify None** (torch 2.10+rocm7.13): `_jit_compile` non gestisce `hipified_path=None` (file lasciati invariati da hipify) → crash. **Monkeypatch** prima di `load()`: avvolgere `torch.utils.hipify.hipify_python.hipify` e sostituire ogni `None` col path sorgente originale (`.cu` valido; `_is_cuda_file` accetta `.cu` e `.hip`).
4. **Device libs**: il bitcode HIP SDK 7.2 sta in `C:\HIP-SDK\lib\llvm\amdgcn\bitcode` (non sotto la root) → passare `extra_cuda_cflags=[r"--rocm-device-lib-path=C:\HIP-SDK\lib\llvm\amdgcn\bitcode"]`.
5. `build_directory` corto (es. `C:\vw_p0build`). Link finale: MSVC `link.exe` con `c10_hip.lib torch_hip.lib amdhip64.lib`.

> Nota: vLLM reale usa **CMake** (non `cpp_extension`), quindi il bug hipify-None non si applica lì; ma le lezioni `--rocm-device-lib-path` e ambiente MSVC valgono. Prossima escalation: un kernel reale (norm/paged-attention), poi un subset di `csrc`.

## Altri esperimenti Fase 0
- [x] `torch.distributed` → **CONFERMATO ASSENTE** (`is_available()==False`) → short-circuit init obbligatorio.
- [x] Kernel HIP reali su gfx1100: **RMSNorm** (max_err ~1e-6), **rocWMMA fp16 GEMM** (rel 0.0), **rocWMMA INT8 GEMM** (diff 0) — `experiments/phase0_kernels`.
- [x] Triton complesso: **`tl.dot` matmul** (rel 0.0, → WMMA) + **softmax** (err ~4e-9) — `experiments/phase0_triton`. Oltre il vector-add. - [ ] `hipBLASLt` fp16/bf16 attivo su gfx1100 o fallback hipBLAS?
- [ ] Compila il kernel **W4A16 (PR #41394)** vLLM reale sotto questo toolchain? (richiede checkout vLLM; building block INT8 WMMA già validato)
- [ ] **Flash-attention** Triton/HIP a head_size=128 + paged KV + GQA (il vero gate del blocker attention).
