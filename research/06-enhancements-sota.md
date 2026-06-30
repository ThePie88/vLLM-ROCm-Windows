# 06 — Enhancement SOTA (oltre la parità)

Obiettivo: *superare* vLLM baseline su gfx1100, non solo replicarlo. Ricerca multi-agente + verifica avversariale. Tutto calibrato sul nostro stack (gfx1100, 20 GB, Triton nativo, build HIP, **no FP8 hardware**, **LDS 64 KB/workgroup**).

## Il "metodo cinese, controparte di TurboQuant" — RISOLTO: **KVarN** (Huawei)

Identificato dall'utente e confermato via web (2026-06-29). **KVarN** = *"Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks"*, **arXiv:2606.03458 (giu 2026)**, **Huawei CSL**, Apache-2.0, repo `huawei-csl/KVarN`. (Non l'avevo nella prima ricerca perché uscito da pochissimo.)

- **Tecnica:** calibration-free; pipeline cache → **rotazione Hadamard** (sparge gli outlier) → **normalizzazione varianza iterativa** (Sinkhorn-like, alterna std colonna/riga in log-space) → **quant asimmetrica low-bit** (scale ri-applicate in lettura: keys per-canale, values per-token). Preset `kvarn_k4v2` (4-bit K / 2-bit V), tile `g128`/`g64`.
- **vs TurboQuant (i numeri):** su Qwen3-32B (AIME25, 16K, TP=2) KVarN eguaglia FP16 a **~4× capacità KV**; TurboQuant riporta **40–52% throughput in meno** per 2.3–3.7× capacità. KVarN dà **~1.3× il throughput di FP16** ⇒ **fino a ~2.4× il throughput di TurboQuant a pari capacità, con accuratezza superiore.** Costruito su KIVI (risolve il "token scaling problem").
- **Perché è IDEALE per noi:** **kernel puramente Triton (JIT a runtime)** — niente CUDA-only; **int4/int2 + compute float16 → nessun FP8 hardware** (vincolo RDNA3 rispettato); **backend KV-cache nativo vLLM, un flag** (`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128 --dtype float16`); primo backend vLLM sub-8-bit a supportare **modelli MLA** (latent KV → int4); **3–5× contesto** (cruciale a 20 GB).
- **Caveat:** fork di **vLLM v0.23.0** (probabile torch 2.11; noi 2.10 → da riconciliare) e **nessuna menzione ROCm/AMD** (NVIDIA-focused) — ma essendo Triton+float16+int, è esattamente la categoria che abbiamo de-riskato su gfx1100. Da provare: i kernel Triton di KVarN compilano/girano corretti sotto triton-windows su gfx1100.

> Nota storica: la mia ipotesi precedente (VecInfer/MILLION/RotateKV) era nella nicchia giusta ma sbagliata; la risposta vera è **KVarN**.

## DeepSeek attention — verdetto per gfx1100

| Metodo | Cos'è | Per noi? | Path kernel gfx1100 |
|---|---|---|---|
| **MLA** (Multi-head Latent Attention, V2) | compressione KV low-rank, cache ~576 dim vs ~8K → **~7% della MHA (~14×)** | **SÌ, win a basso rischio** — solo se servi modelli MLA (famiglia DeepSeek) | Backend **`TRITON_MLA`** nativo, stesso source NV/AMD/Intel → gira su gfx1100 oggi, serve solo tuning. AITER-MLA = CDNA-only (evita) |
| **NSA** (Native Sparse Attention, ACL 2025 Best Paper) | sparsità gerarchica dinamica (compress + top-k + sliding-window), trainable | **FORSE** — il target sparse realistico | **Kernel Triton fusi ufficiali esistono** (fla-org/native-sparse-attention); Triton-feasible su gfx1100 ma serve **retiling per il cap LDS 64 KB**. Research-grade |
| **DSA** (DeepSeek Sparse Attention, V3.2) | lightning indexer + top-k=2048, O(L²)→O(Lk) | **NO (per ora)** | Path produzione **CUDA-only** (DeepGEMM+FlashMLA+TileLang); port ROCm bloccato — il kernel chiede **96 KB** LDS e già crasha su MI325X (64 KB). gfx1100 stesso limite → da-zero. Defer |

**Bottom line:** MLA via `TRITON_MLA` = pronto/basso rischio (se servi modelli MLA). NSA = sparse Triton-feasible ma retiling 64 KB. DSA = research-grade su RDNA3, rimandare.

## TurboQuant — verdetto

Identità **confermata**: Google Research + NYU + DeepMind, arXiv:2504.19874, ICLR 2026. Quantizzatore **KV-cache** calibration-free (rotazione random + scalar quant + residuo QJL 1-bit), ~4-6× compressione KV, ~8× speedup logit su H100. **Non cinese.**

Fatti decisivi: (a) **vLLM stesso ha misurato che FP8-KV batte TurboQuant** su throughput/latenza a pari accuratezza; (b) novità contestata (RaBitQ reuse); (c) niente codice ufficiale, PR vLLM NVIDIA-only; (d) **già portato su RDNA3 Linux** (llama.cpp HIP su 7900 XTX; AMD lo ha productionizzato su MI355X) — **mai su Windows nativo**.
→ **Priorità P2** (TurboQuant in sé). Ma poiché su gfx1100 **non c'è FP8 hardware**, la mossa KV migliore è una quant int rotation-based in Triton — e il candidato concreto è **KVarN** (vedi sopra), che è esattamente questo (Hadamard + varianza + int4/int2, kernel Triton), batte TurboQuant, ed è già un backend vLLM. KVarN > TurboQuant per il nostro stack.

## Enhancement backlog (prioritizzato)

### P0 — fai subito (nessun kernel custom, o engine già funzionante)
| Metodo | Cosa dà | Costo sul nostro stack |
|---|---|---|
| **W4A16 via kernel HIP nativo (PR #41394)** | 4-bit pesi, ~3-4× VRAM; 2.5-4.2× vs vecchio Triton; accelera GPTQ+AWQ | **config** — kernel merged 2026-05-29, **mira esplicitamente gfx1100** (WMMA bf16 + v_dot2). **verificare che compili sotto HIP SDK 7.2 nativo Windows**; capire quale tag/wheel lo include |
| **Prefix caching automatico + chunked prefill** | TTFT/throughput gratis | **none** — default-on V1, scheduler-level |
| **Speculative decoding N-gram / prompt-lookup** | fino ~2.8× su RAG/code; zero VRAM extra | **none** — flag, proposer CPU-side. Ideale su scheda VRAM-tight |
| **Backend `TRITON_ATTN` (tuning + graph capture)** | il vero engine paged-attn per gfx1100 | **Triton** — funzionante; investimento = tuning. "default" version-fragile (#39965) → pinna versione, set `VLLM_ATTENTION_BACKEND` esplicito |

### P1
- **KVarN KV-quant (Huawei, vLLM-native)** — 4-bit K / 2-bit V, calibration-free, **~4× capacità KV, throughput > FP16, accuratezza FP16, batte TurboQuant ~2.4×**. **Kernel Triton + float16 + int (no FP8)** → fit ideale RDNA3. Integrazione = un flag. Costo: portare i kernel Triton di KVarN su triton-windows/gfx1100 + riconciliare il fork vLLM v0.23.0 con torch 2.10. **Forte candidato per il miglior win KV-cache su 20 GB.**
- **GGUF Q4_K_M/Q5_K_M ROCm** (~3-4× VRAM, copertura ampia, escape hatch quando Triton fallisce; 62 tok/s su 7900 XTX) — none.
- **PagedAttention + hybrid KV manager** (<4% waste, sblocca long-ctx in 20 GB) — none.
- **EAGLE-3 speculative** (~2.5× latenza single-stream) — config + VRAM draft-head; verificare che il tree-verify non colpisca kernel CUDA-only.
- **MLA via `TRITON_MLA`** — Triton, solo se servi modelli MLA.

### P2 (custom-kernel / research-grade / track)
- **INT8 W8A8 GEMM custom** (iu8 WMMA reale su RDNA3) — *miglior opportunità custom-kernel*; gate: verificare che triton-windows 3.6 emetta `v_wmma_i32_16x16x16_iu8` su gfx1100.
- **INT8 KV cache custom** (~2× memoria KV, evita il crash FP8-KV+prefix #13147).
- **RotateKV / Hadamard 2-bit KV** (long-ctx; Hadamard cheap, Triton-feasible).
- **TurboQuant KV**, **NSA**, **CommVQ** (2-bit additive VQ, decode-by-matmul RDNA3-friendly) — track.
- **FP8 e4m3 KV** — solo capacità (no compute), **crasha con prefix caching** (#13147) → fragile.
- **hipGraph FULL capture** — hang documentati (#39010) → partire `--enforce-eager` → PIECEWISE.
- **CPU KV-offload/LMCache** — solo al muro VRAM.

### DA EVITARE (CUDA/CDNA-only, nessun path ROCm)
Marlin/Machete (PTX/Hopper) · FlashInfer-ROCm & FA3 (AITER/CDNA, Hopper TMA/WGMMA) · AQLM/QuIP#/VPTQ/QTIP (decode CUDA-only; ExLlamaV3/QTIP marca ROCm unsupported) · FP8/MXFP4 weights (no hardware RDNA3 → fallback a non-quantizzato).

## Stack quantizzazione consigliato (RX 7900 XT, 20 GB)
- **Primario**: **W4A16 via kernel HIP nativo (PR #41394)** con checkpoint AWQ/GPTQ 4-bit (group-128/32). 27-32B in ~16-18 GB; 13-14B in ~8-9 GB. *Gate: compila su Windows HIP SDK 7.2?*
- **KV cache**: **FP16 di default** (niente FP8 su RDNA3). Per più contesto → **INT8 KV custom** (iu8 reale) o **2-bit rotation (RotateKV/Hadamard)** in Triton — **non** FP8.
- **Fallback/portabilità**: **GGUF Q4_K_M/Q5_K_M** ROCm.
- **Miglior scommessa custom**: **Triton INT8 W8A8 tunato per WMMA iu8 gfx1100**.

## Caveat RDNA3 portanti (riepilogo)
1. **Niente FP8/MXFP4 matmul** (solo FP16/BF16/IU8/IU4). 2. **FP8 KV + prefix caching CRASHA** (#13147). 3. **LDS 64 KB/workgroup** (= MI325X; kernel DeepSeek DSA da 96 KB non girano). 4. **AITER non gira affatto su RDNA** → **Triton è l'engine**. 5. CDNA-only da evitare (sopra). 6. **FA2 forward esiste su RDNA3** via CK (backward no = solo training). 7. `TRITON_ATTN` default version-fragile → pinna. 8. **Windows nativo è il vero rischio**: tutta l'evidenza AMD esistente è Linux; budget tempo per debug Triton+toolchain.

## Gate/Domande aperte chiave
- **Il kernel W4A16 (PR #41394) compila sotto Windows nativo HIP SDK 7.2 / clang 22 + MSVC?** (unknown più importante per il path quant primario)
- triton-windows 3.6 emette l'intrinsic **INT8 WMMA `iu8`** su gfx1100, o solo FP16/BF16? (gate per INT8 custom)
- EAGLE-3: il tree-verify V1 passa pulito da `TRITON_ATTN` su gfx1100 o colpisce un kernel CUDA-only?
- NSA Triton kernels ritilabili entro 64 KB LDS mantenendo l'online top-k?
