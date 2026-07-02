"""Flash-layout decode kernel: correctness vs fp32 reference (incl. sliding) + timing.
Cache is TRITON_ATTN's flash layout kc/vc [num_blocks, block_size, num_kv_heads, head_size]."""
import os, math, time
for _d in (r"C:\HIP-SDK\bin", r"C:\HIP-SDK\lib", r"C:\vw_attnflash_build"):
    try: os.add_dll_directory(_d)
    except Exception: pass
import torch
torch.ops.load_library(r"C:\vw_attnflash_build\vllm_win_attn_flash_C.pyd")
assert hasattr(torch.ops._C, "paged_attention_flash")
print("flash pyd loaded")

dev = "cuda"
BS = 16
torch.manual_seed(0)

def build_flash(K, V, seq_len, nkv, head):
    nblk = (seq_len + BS - 1) // BS
    kc = torch.zeros(nblk, BS, nkv, head, device=dev, dtype=K.dtype)
    vc = torch.zeros(nblk, BS, nkv, head, device=dev, dtype=K.dtype)
    for t in range(seq_len):
        kc[t // BS, t % BS] = K[t]
        vc[t // BS, t % BS] = V[t]
    return kc, vc, nblk

def ref(q, K, V, nq, nkv, head, seq_len, W):
    grp = nq // nkv
    qf, Kf, Vf = q.float(), K.float(), V.float()
    scale = 1.0 / math.sqrt(head)
    o = torch.empty(nq, head, device=dev, dtype=torch.float32)
    pos = torch.arange(seq_len, device=dev)
    for h in range(nq):
        kvh = h // grp
        s = (qf[h] @ Kf[:, kvh, :].T) * scale
        if W > 0:
            s = s.masked_fill((seq_len - 1 - pos) >= W, float("-inf"))
        o[h] = torch.softmax(s, dim=-1) @ Vf[:, kvh, :]
    return o

for (nq, nkv, head, dt) in [(20, 4, 128, torch.bfloat16), (16, 8, 256, torch.float16), (20, 4, 128, torch.float16)]:
    for (seq_len, W) in [(300, 0), (1000, 0), (1000, 256), (2000, 512)]:
        q = torch.randn(nq, head, device=dev, dtype=dt) * 0.4
        K = torch.randn(seq_len, nkv, head, device=dev, dtype=dt) * 0.4
        V = torch.randn(seq_len, nkv, head, device=dev, dtype=dt) * 0.4
        kc, vc, nblk = build_flash(K, V, seq_len, nkv, head)
        out = torch.empty(1, nq, head, device=dev, dtype=dt)
        btab = torch.arange(nblk, device=dev, dtype=torch.int32).view(1, nblk)
        slens = torch.tensor([seq_len], device=dev, dtype=torch.int32)
        scale = 1.0 / math.sqrt(head)
        torch.ops._C.paged_attention_flash(out, q.view(1, nq, head), kc, vc, nkv, scale,
                                            btab, slens, BS, seq_len, W)
        on = out.view(nq, head).float()
        orf = ref(q, K, V, nq, nkv, head, seq_len, W)
        rel = (on - orf).abs().max() / (orf.abs().max() + 1e-6)
        tag = f"head={head} {str(dt)[6:]} nq/nkv={nq}/{nkv} seq={seq_len} W={W}"
        print(f"{tag:52s} rel={rel.item():.3e}  {'OK' if rel < 2e-2 else 'MISMATCH'}")

# timing (bf16 head 128, ERNIE-like)
print("---- timing (bf16 head128 nq20 nkv4) ----")
nq, nkv, head, dt = 20, 4, 128, torch.bfloat16
for seq_len in (512, 1024, 2048):
    nblk = (seq_len + BS - 1) // BS
    q = torch.randn(1, nq, head, device=dev, dtype=dt)
    out = torch.empty(1, nq, head, device=dev, dtype=dt)
    kc = torch.randn(nblk, BS, nkv, head, device=dev, dtype=dt)
    vc = torch.randn(nblk, BS, nkv, head, device=dev, dtype=dt)
    btab = torch.arange(nblk, device=dev, dtype=torch.int32).view(1, nblk)
    slens = torch.tensor([seq_len], device=dev, dtype=torch.int32)
    scale = 1.0 / math.sqrt(head)
    def run():
        torch.ops._C.paged_attention_flash(out, q, kc, vc, nkv, scale, btab, slens, BS, seq_len, 0)
    for _ in range(50): run()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(300): run()
    torch.cuda.synchronize()
    print(f"FLASH seq={seq_len:5d}: {(time.perf_counter()-t0)/300*1e6:7.1f} us")
print("MB_FLASH_DONE")
