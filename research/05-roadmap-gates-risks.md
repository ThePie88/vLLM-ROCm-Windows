# 05 — Roadmap, gate go/no-go & rischi

Target v1: **single-GPU gfx1100 (RX 7900 XT), Windows nativo, parità di comportamento col path CUDA di riferimento**. Niente WSL2, niente offload CPU "inventato" (solo i meccanismi nativi di vLLM).

## Fase 0 — Ambiente & prova della fondazione *(parzialmente già superata)*

Obiettivo: confermare che il progetto non è morto in partenza.
- [x] `torch+rocm` gira su gfx1100 (cuda.is_available, fp16 matmul, SDPA). **fatto**
- [x] Triton compila+esegue un kernel nativo su gfx1100. **fatto** (base)
- [x] Toolchain HIP+MSVC+WinSDK presente. **fatto**
- [x] **Build+import di una `.pyd` HIP multi-file** via `torch.utils.cpp_extension`. **SUPERATO (2026-06-29)** — kernel su gfx1100, `max_err 0.0`. Ricetta in doc 03.
- [x] Presenza `torch.distributed` nel wheel → **CONFERMATO ASSENTE** (`is_available()==False`). Strategia: short-circuit incondizionato init a `world_size==1`.
- [ ] Decidere il pin versione (driver+ROCm+torch 2.10+triton) e il tag vLLM compatibile (gap 2.10 vs 2.11).

**GO/NO-GO**: `torch` gira **E** una `.pyd` HIP custom builda+importa+lancia un kernel. Se la `.pyd` non si costruisce/importa → ripiego "Triton-only / SDPA-only" (ri-scope duro).

## Fase 1 — Inferenza eager single-GPU, kernel stubbati (~3-5 settimane)

Obiettivo: vLLM importa e **genera token** su gfx1100, anche lento, con `TORCH_SDPA`.
- Patch `setup.py` per win32+hip; installa vLLM core con kernel stubbati.
- **Platform plugin out-of-tree** (`vllm.platform_plugins`): detection via `torch.cuda` (no `amdsmi`), `world_size==1` short-circuit, FP8/AITER/custom-AR off.
- Forza `enforce_eager=True`, `tensor_parallel_size=1`, backend `TORCH_SDPA`.
- Modello dense fp16/bf16 piccolo (es. Llama-3.2-1B/3B) end-to-end; **validazione output vs reference CUDA/Linux**.

**GO/NO-GO**: token corretti su un gfx1100 in eager mode. → **SUPERATO (2026-06-30)**: vLLM v0.19.1 genera testo coerente su gfx1100 (OPT-125m, eager, `TRITON_ATTN`, single-GPU) via plugin out-of-tree `windows_rocm_plugin` + shim single-process di `torch.distributed`. Ricetta completa: `run/README.md` e memoria `phase1-vllm-first-token`.

## Fase 2 — Kernel attention / paged-KV (fase critica, ~6-10 settimane)

Obiettivo: un vero backend paged-attention, non SDPA.
- Bake-off: (a) `TRITON_ATTN` via triton-windows; (b) port del kernel HIP `__GFX11__` WMMA; baseline `TORCH_SDPA`.
- Validazione numerica a head_size=128 + paged KV + GQA; misura throughput prefill+decode.
- Aggiungi AWQ/GPTQ (Triton) per modelli dense se Triton regge; preferisci AWQ dense.

**GO/NO-GO**: almeno una rotta paged-attention **corretta E più veloce di SDPA** su gfx1100. Se nessuna → si rilascia solo backend SDPA eager (corretto ma lento) = milestone "world-first" non competitiva.

## Fase 3 — Ottimizzazione / perf (open-ended)

- Riabilita HIP graphs se stabile su HIP-Windows; tuning occupancy/LDS wave32 (entro 64 KB); valida `hipBLASLt` fp16/bf16 (altrimenti hipBLAS).
- Espandi copertura quantizzata (correttezza `gptq_gemm`, evita limite MoE SiLU-only).
- **Enhancement SOTA** oltre la parità (vedi `06-enhancements-sota.md`): MLA (basso rischio), poi NSA/sparse, KV-quant.

**GO/NO-GO (successo di progetto)**: gfx1100 serve un modello reale a throughput entro una frazione usabile di llama.cpp/Vulkan; altrimenti = artefatto di ricerca.

## Rischi (ordinati)

1. **Attention fallisce su entrambe le rotte** (massimo). Mitigazione: bake-off Fase 2 con `TORCH_SDPA` come floor garantito.
2. **Build `.pyd` HIP multi-file non provato a scala vLLM**. Mitigazione: PoC triviale in Fase 0 prima di impegnarsi (Gate A).
3. **"L'intero stack ROCm non è ancora su Windows"** (parole AMD): librerie userspace (hipBLASLt tunato, comm libs) incomplete. Mitigazione: TheRock per i pezzi mancanti; restare alle lib confermate presenti.
4. **Gap versione torch (2.10 vs 2.11 richiesto da vLLM main)**. Mitigazione: pinnare un tag vLLM compatibile con 2.10 o rilassare l'assert (preferito a buildare torch da source).
5. **`torch.distributed` assente dal wheel Windows** — **CONFERMATO** (`is_available()==False`, ROCm #5689). Mitigazione: short-circuit incondizionato dell'init a `world_size==1` (non basta affidarsi al fallback gloo: non c'è).
6. **Sensibilità driver/SDK**: hang di compilazione Triton dopo update driver; rename DLL 7.1.1↔7.2. Mitigazione: **pin** driver + una sola versione ROCm/HIP.
7. **Bug DPP -O0 su gfx1100** (permanente). Mitigazione: solo Release/-O2+; audit kernel RDNA3 per intrinsics DPP/cross-lane.
8. **Scope creep su GPU non supportate** (gfx1102/1103). Mitigazione: v1 ufficialmente solo gfx1100.
9. **Nessun prior art** native-Windows-ROCm-vLLM. Tutto empirico. (`SystemPanic/vllm-windows` prova solo il path CUDA.)

## Domande aperte per il proprietario di progetto
1. ~~GPU target?~~ → **RX 7900 XT / gfx1100** (confermato).
2. ~~WSL2 fallback?~~ → **No, Windows nativo** (confermato).
3. Accettiamo lo **stack pinnato** (driver + ROCm 7.13 + torch 2.10 + triton 3.6 + Python 3.12) e un **tag vLLM compatibile con torch 2.10** invece di inseguire `main` (che vuole 2.11)?
4. Una **v1 eager / potenzialmente SDPA-only** è una milestone "world-first" accettabile, o il throughput competitivo è requisito duro (alza l'asticella a un attention WMMA/Triton funzionante)?
