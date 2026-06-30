"""Phase-2 benchmark harness: measure tok/s (prefill TTFT + decode) and VRAM on gfx1100.

Run from run/ (avoids the vllm/ clone-dir shadowing). Configure via env:
  VLLM_BENCH_MODEL   (default facebook/opt-125m)
  VLLM_BENCH_MAXTOK  (default 128)   decode tokens
  VLLM_BENCH_PROMPTS (default 1)     batch size
  VLLM_BENCH_MAXLEN  (default 2048)
  VLLM_BENCH_GPUUTIL (default 0.6)
  VLLM_BENCH_BACKEND (default TRITON_ATTN)
  VLLM_BENCH_QUANT   (optional, e.g. awq / gptq)
"""
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
# torch.distributed.tensor is broken on this USE_DISTRIBUTED=0 build; conch/other kernels
# may trigger torch.compile/inductor which imports it. enforce_eager doesn't need compile.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_BENCH_MODEL", "facebook/opt-125m")
MAXTOK = int(os.environ.get("VLLM_BENCH_MAXTOK", "128"))
PROMPTS = int(os.environ.get("VLLM_BENCH_PROMPTS", "1"))
MAXLEN = int(os.environ.get("VLLM_BENCH_MAXLEN", "2048"))
GPUUTIL = float(os.environ.get("VLLM_BENCH_GPUUTIL", "0.6"))
BACKEND = os.environ.get("VLLM_BENCH_BACKEND", "TRITON_ATTN")
QUANT = os.environ.get("VLLM_BENCH_QUANT") or None
GIB = 2 ** 30

torch.cuda.reset_peak_memory_stats()
free0, total = torch.cuda.mem_get_info()
print(f"== {MODEL} | backend={BACKEND} quant={QUANT} | GPU {total/GIB:.1f} GiB, free {free0/GIB:.1f} GiB")

GRAPH = os.environ.get("VLLM_BENCH_GRAPH", "0") == "1"
llm_kwargs = dict(
    model=MODEL, dtype="float16", attention_backend=BACKEND, quantization=QUANT,
    tensor_parallel_size=1, gpu_memory_utilization=GPUUTIL, max_model_len=MAXLEN,
    trust_remote_code=(os.environ.get("VLLM_BENCH_TRUST", "1") == "1"),
)
if GRAPH:
    # FULL_DECODE_ONLY captures a full HIP graph for decode with NO inductor (mode=NONE),
    # collapsing the eager per-op dispatch overhead. PIECEWISE would need inductor (broken).
    llm_kwargs["enforce_eager"] = False
    llm_kwargs["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    print("== cudagraph: FULL_DECODE_ONLY (mode=NONE, no inductor)")
else:
    llm_kwargs["enforce_eager"] = True

t_load = time.perf_counter()
llm = LLM(**llm_kwargs)
print(f"engine init: {time.perf_counter()-t_load:.1f}s")
free1, _ = torch.cuda.mem_get_info()
print(f"VRAM used after init: {(total-free1)/GIB:.2f} GiB (free {free1/GIB:.2f})")

prompts = ["Write a detailed essay about the history and future of GPU computing."] * PROMPTS
sp_warm = SamplingParams(temperature=0.0, max_tokens=8)
sp_one = SamplingParams(temperature=0.0, max_tokens=1)
sp_full = SamplingParams(temperature=0.0, max_tokens=MAXTOK, ignore_eos=True)

llm.generate(prompts, sp_warm)  # warmup (compile triton kernels etc.)
torch.cuda.synchronize()

t0 = time.perf_counter()
llm.generate(prompts, sp_one)
torch.cuda.synchronize()
ttft = (time.perf_counter() - t0) / max(PROMPTS, 1)

t0 = time.perf_counter()
out = llm.generate(prompts, sp_full)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
gen = sum(len(o.outputs[0].token_ids) for o in out)

print("-" * 50)
print(f"prompts={PROMPTS}  decode_max_tokens={MAXTOK}")
print(f"~TTFT (single-token call): {ttft*1000:.0f} ms")
print(f"decode: {gen} tok in {dt:.2f}s  ->  {gen/dt:.1f} tok/s aggregate, {gen/dt/max(PROMPTS,1):.1f} tok/s/req")
print(f"torch peak allocated: {torch.cuda.max_memory_allocated()/GIB:.2f} GiB")
free2, _ = torch.cuda.mem_get_info()
print(f"VRAM used (device) at end: {(total-free2)/GIB:.2f} GiB")
print("BENCH_OK")
