"""S5 microbench PART A: native _C.paged_attention_v1 alone (no vllm import -> no _C namespace conflict).
gemma sliding-layer decode: 1 query token, head_size 256, 16 q / 8 kv heads, context SEQ, block_size 16.
Native v0 paged KV layout. Prints us/call for varying context."""
import os, time
for _d in (r"C:\HIP-SDK\bin", r"C:\HIP-SDK\lib", r"C:\vw_attn_build"):
    try: os.add_dll_directory(_d)
    except Exception: pass
import torch
torch.ops.load_library(r"C:\vw_attn_build\vllm_win_attn_C.pyd")
assert hasattr(torch.ops._C, "paged_attention_v1"), "no paged_attention_v1"
print("native pyd loaded")

dev, dt = "cuda", torch.float16
NQ, NKV, HD, BS = 16, 8, 256, 16
X = 16 // dt.itemsize
scale = 1.0 / (HD ** 0.5)
one = torch.tensor(1.0, device=dev, dtype=torch.float32)

def bench(fn, it=300, wu=60):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6

for SEQ in (512, 1024, 2048, 4096):
    NBLK = (SEQ + BS - 1) // BS
    q = torch.randn(1, NQ, HD, device=dev, dtype=dt)
    out_n = torch.empty(1, NQ, HD, device=dev, dtype=dt)
    kc = torch.randn(NBLK, NKV, HD // X, BS, X, device=dev, dtype=dt)
    vc = torch.randn(NBLK, NKV, HD, BS, device=dev, dtype=dt)
    btab = torch.arange(NBLK, device=dev, dtype=torch.int32).view(1, NBLK)
    slens = torch.tensor([SEQ], device=dev, dtype=torch.int32)
    W = int(os.environ.get("MB_SW", "0"))
    def run():
        torch.ops._C.paged_attention_v1(out_n, q, kc, vc, NKV, scale, btab, slens,
            BS, SEQ, None, "auto", one, one, 0, 0, 0, 0, 0, W)
    try:
        run(); t = bench(run)
        print(f"NATIVE seq={SEQ:5d}: {t:7.1f} us")
    except Exception as e:
        print(f"NATIVE seq={SEQ:5d} FAILED: {repr(e)[:160]}")
print("MB_NATIVE_DONE")
