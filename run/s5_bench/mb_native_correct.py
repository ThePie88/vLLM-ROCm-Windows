"""S5 correctness: does native _C.paged_attention_v1 produce CORRECT output at head_size 256, GQA 16q/8kv?
Build logical Q,K,V; pack K,V into the v0 paged layout; run native; compare vs a reference softmax(QK*scale)V.
No sliding mask here (full causal over the whole stored context = plain full attention for 1 query)."""
import os, math
for _d in (r"C:\HIP-SDK\bin", r"C:\HIP-SDK\lib", r"C:\vw_attn_build"):
    try: os.add_dll_directory(_d)
    except Exception: pass
import torch
torch.ops.load_library(r"C:\vw_attn_build\vllm_win_attn_C.pyd")
print("native pyd loaded")

dev, dt = "cuda", torch.float16
NQ, NKV, HD, BS = 16, 8, 256, 16
GRP = NQ // NKV
scale = 1.0 / math.sqrt(HD)
torch.manual_seed(0)

for SEQ in (48, 512, 1023):
    NBLK = (SEQ + BS - 1) // BS
    X = 16 // dt.itemsize
    q = torch.randn(NQ, HD, device=dev, dtype=dt) * 0.5
    K = torch.randn(SEQ, NKV, HD, device=dev, dtype=dt) * 0.5
    V = torch.randn(SEQ, NKV, HD, device=dev, dtype=dt) * 0.5

    # reference (fp32)
    qf, Kf, Vf = q.float(), K.float(), V.float()
    out_ref = torch.empty(NQ, HD, device=dev, dtype=torch.float32)
    for h in range(NQ):
        kvh = h // GRP
        s = (qf[h] @ Kf[:, kvh, :].T) * scale          # [SEQ]
        p = torch.softmax(s, dim=-1)
        out_ref[h] = p @ Vf[:, kvh, :]

    # pack v0 paged layout
    kc = torch.zeros(NBLK, NKV, HD // X, BS, X, device=dev, dtype=dt)
    vc = torch.zeros(NBLK, NKV, HD, BS, device=dev, dtype=dt)
    for t in range(SEQ):
        b, off = t // BS, t % BS
        for kvh in range(NKV):
            kc[b, kvh, :, off, :] = K[t, kvh, :].view(HD // X, X)
            vc[b, kvh, :, off] = V[t, kvh, :]
    q_in = q.view(1, NQ, HD)
    out_n = torch.empty(1, NQ, HD, device=dev, dtype=dt)
    btab = torch.arange(NBLK, device=dev, dtype=torch.int32).view(1, NBLK)
    slens = torch.tensor([SEQ], device=dev, dtype=torch.int32)
    one = torch.tensor(1.0, device=dev, dtype=torch.float32)
    torch.ops._C.paged_attention_v1(out_n, q_in, kc, vc, NKV, scale, btab, slens,
        BS, SEQ, None, "auto", one, one, 0, 0, 0, 0, 0)

    on = out_n.view(NQ, HD).float()
    err = (on - out_ref).abs()
    rel = err.max() / (out_ref.abs().max() + 1e-6)
    print(f"seq={SEQ:5d}: max_abs_err={err.max().item():.4e} mean_abs={err.mean().item():.4e} "
          f"rel={rel.item():.4e}  {'OK' if rel < 2e-2 else 'MISMATCH'}")
print("MB_CORRECT_DONE")
