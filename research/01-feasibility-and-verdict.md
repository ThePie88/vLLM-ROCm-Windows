# 01 — Feasibility & verdetto

## Verdetto

**REALISTICAMENTE FATTIBILE** per **inferenza single-GPU su gfx1100** (RX 7900 XT/XTX), come sforzo ingegneristico da primo-al-mondo. La v1 realistica girerà **via Triton e/o un'estensione HIP di attention portata a mano, inizialmente in eager mode** — non un build 1:1 di vLLM upstream. **Multi-GPU = fuori scope** (niente RCCL su Windows). gfx1102/gfx1103 (RX 7600, APU) = sperimentali/fuori scope ufficiale Windows.

Il progetto **dipende interamente dal layer PyTorch-ROCm su Windows nativo**, e quel layer **sul nostro hardware funziona** (misurato): è ciò che ribalta il quadro da "bloccato" a "fattibile".

## Ground-truth (misurato 2026-06-29) vs ricerca web

La ricerca web era prudente/pessimista su due punti che **i probe sul ferro hanno risolto in positivo** per il nostro box specifico:

| Tema | Ricerca web (prudente) | Misura sul nostro gfx1100 | Esito |
|---|---|---|---|
| PyTorch+ROCm su Windows per gfx1100 | "preview/non provato; ufficiale solo 2.9.1+rocm7.2.1" | **`torch 2.10.0+rocm7.13.0`, `cuda.is_available()=True`, fp16 matmul OK, SDPA OK** | Risolto (stack TheRock più nuovo dell'ufficiale) |
| Triton nativo Windows ROCm | "sperimentale, solo via ZLUDA, leaning negative" | **`triton-windows 3.6` compila+esegue un kernel nativo, err 0.0 (no ZLUDA)** | Codegen base risolto; attention-scale da validare |
| Gap versione torch | "2.9.1 vs 2.11 richiesto da vLLM" | Abbiamo **2.10.0** → gap di 1 minor | Ridotto |
| Toolchain build HIP nativa | "HIP SDK Windows parziale, CMake rotto" | HIP SDK 7.2 (clang 22) + MSVC 14.44 + Win SDK 10.0.26100 **tutto presente** | Disponibile (build da provare) |

> Nota importante: il nostro stack è **TheRock-class `rocm7.13`**, non la "PyTorch on Windows Edition 7.2.1" ufficiale. È più capace ma "nightly/preview" — quindi va **pinnato** (driver + ROCm + torch + triton) e trattato come dipendenza fissa, non mobile.

## Critical path (ogni gate sblocca il successivo)

- **Gate A — Build & import di una `.pyd` HIP multi-file** via `torch.utils.cpp_extension` su gfx1100 nativo. → **SUPERATO (2026-06-29)**: kernel `.cu` compilato da hipcc/clang 22 (offload gfx1100-1103) + binding linkato da MSVC `link.exe` contro torch 2.10+rocm7.13 → `.pyd` importato, kernel eseguito su GPU, `max_err 0.0`. *Nessun precedente pubblico.* **La tesi delle estensioni native è dimostrata.** Ricetta in [03](03-environment-toolchain-build.md).
- **Gate B — vLLM core importabile/eseguibile su Windows**: patch `VLLM_TARGET_DEVICE`, kernel stubbati, `enforce_eager`, `world_size==1`, tramite un **platform plugin out-of-tree** che sostituisce la detection `amdsmi` con `torch.cuda`.
- **Gate C — Path di attention funzionante su gfx1100**: il make-or-break. Bake-off tra (i) `TRITON_ATTN` puro-Triton e (ii) port del kernel HIP `__GFX11__` WMMA di vLLM, con `TORCH_SDPA` come floor di correttezza.

## Il blocker numero uno

**Un path attention / paged-KV corretto e veloce su gfx1100 nativo Windows.** Entrambe le rotte hanno incognite serie:

- **(i) `TRITON_ATTN` puro-Triton** — elegante e portabile; ma le kernel Triton flash-attention hanno fallimenti documentati su gfx1100 *anche su Linux* (vLLM #4514: stack-frame overflow / accuratezza; spesso si forza `VLLM_USE_TRITON_FLASH_ATTN=0`). Il nostro Triton compila un vector-add: **non prova ancora** la correttezza a scala-attention (head_size=128, paged KV, GQA).
- **(ii) Estensione HIP `__GFX11__` WMMA** (`csrc/rocm/attention.cu`) — la logica wave32 esiste già nel source, ma ha compilato **solo su Linux**; su Windows i CMake-config del HIP SDK sono storicamente rotti/non-relocatable e c'è un **bug DPP a -O0 su gfx1100** (build solo Release). Nessun precedente nativo Windows a scala-LLM.

**Go/no-go del progetto = far funzionare almeno UNA delle due su gfx1100 reale (Fase 2).** Floor garantito: `TORCH_SDPA` (corretto ma lento) → accettabile come "world-first" ma non competitivo.

## Cosa NON è più un problema (grazie alla misura)
- "Esiste PyTorch-ROCm su Windows?" → **Sì, e gira sul nostro box.**
- "Triton gira nativo o solo ZLUDA?" → **Gira nativo (codegen base provato).**
- "C'è la toolchain per buildare HIP?" → **Sì** (HIP SDK 7.2 + MSVC + Win SDK).

## Cosa resta genuinamente incerto
- Correttezza+perf delle kernel Triton **a scala attention/MoE** su gfx1100 (oltre il vector-add).
- Build+import di estensioni HIP **multi-file** vLLM-scale (Gate A).
- ~~Presenza di `torch.distributed` nel wheel Windows~~ → **CONFERMATO ASSENTE** sul nostro box (`torch.distributed.is_available()==False`). Impatta l'init anche a `world_size==1` → serve short-circuit incondizionato (vedi doc 04/05).
- Copertura `hipBLASLt` fp16/bf16 tunata per gfx1100 (altrimenti fallback `hipBLAS` più lento).
- Perf reale: anche su Linux vLLM su RDNA3 è sotto llama.cpp (coerente col fatto che RDNA3 WMMA = throughput DOT, niente matrix-core).
