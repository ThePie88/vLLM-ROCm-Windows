"""Fast M=1 W4A16 dequant-GEMV for single-stream decode on RDNA3, wired as an MPLinearKernel
registered ahead of conch in vLLM's ROCm kernel priority.

Why: on ROCm/Windows, AWQ uint4 has no fast kernel -- vLLM routes AWQ -> awq_marlin ->
choose_mp_linear_kernel([Conch, Exllama]); Exllama rejects uint4 (only uint4b8), Marlin is
CUDA-only, so ONLY conch remains, which runs a throughput tile (block_m=128, tl.dot needs >=16
rows) for a single decode row -> ~21x off bandwidth at M=1. This is a true GEMV (reduction, no
tl.dot / split-K / atomicAdd / zero-init).

Layout: vLLM's awq_marlin path delivers weights in a packed-along-K layout. We REUSE conch's
process_weights_after_loading to normalize (and conch itself for the M>1 prefill path), so this
kernel consumes conch's exact post-process layout:
  w_q : [K//8, N]   int32, packed along K (8 K-values per int32, straight order shift=(k%8)*4)
  w_s : [K//G, N]   fp16  group scales
  w_zp: [K//G, N]   uint8  UNPACKED zero points
Dequant (conch SYMMETRIC_WITH_SHIFT, weight_bias=0 for uint4): w[k,n] = (q(k,n) - z(g,n))*s(g,n).
Validated by token-agreement vs conch on the real model (run/precision_check.py).
"""
import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn}, num_warps=nw)
             for bn in (8, 16, 32) for nw in (1, 2)],
    key=["K", "N"],
)
@triton.jit
def _gemv_k_kernel(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    K, N,
    BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr,   # quant group size (== K rows processed per iteration)
):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)         # output columns
    nmask = n < N
    ROWS: tl.constexpr = GROUP // 8                   # int32 rows per group
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)       # straight order
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    num_groups = K // GROUP
    for g in range(0, num_groups):
        k0 = g * GROUP
        row = (k0 // 8) + tl.arange(0, ROWS)          # int32 rows for this group
        qw = tl.load(qw_ptr + row[:, None] * N + n[None, :],
                     mask=nmask[None, :], other=0)     # [ROWS, BLOCK_N] int32
        # unpack along K: [ROWS, BLOCK_N] -> [ROWS, 8, BLOCK_N] -> [GROUP, BLOCK_N]
        q = (qw[:, None, :] >> shifts[None, :, None]) & 0xF
        q = tl.reshape(q, (GROUP, BLOCK_N)).to(tl.float32)   # k-index = row*8 + nibble

        a = tl.load(a_ptr + k0 + tl.arange(0, GROUP)).to(tl.float32)  # [GROUP]
        contrib = tl.sum(a[:, None] * q, axis=0)      # sum_k a*q  -> [BLOCK_N]
        asum = tl.sum(a)                               # sum_k a (scalar)

        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        # sum_k a*(q - z)*s = s * (sum_k a*q - z*sum_k a)
        acc += (contrib - z * asum) * s

    tl.store(c_ptr + n, acc.to(c_ptr.type.element_ty), mask=nmask)


def gemv_k(a, w_q, w_s, w_zp, group_size):
    # a: [1, K] fp16 ; w_q: [K//8, N] int32 ; w_s: [K//G, N] fp16 ; w_zp: [K//G, N] uint8
    # @triton.autotune picks BLOCK_N/num_warps per (K,N): cache-cold, small-N favors BLOCK_N=8,
    # large-N (gate_up) BLOCK_N=16. Autotune runs during vLLM's eager warmup, before cudagraph capture.
    K = a.shape[-1]
    N = w_s.shape[1]
    c = torch.empty((1, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_k_kernel[grid](a, w_q, w_s, w_zp, c, K, N, GROUP=group_size)
    return c


def _register_kernel_class():
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearKernel,
    )
    from vllm.scalar_type import scalar_types

    class WinRocmAwqGemvKernel(MPLinearKernel):
        """M=1 W4A16 GEMV on conch's normalized layout; conch handles weight prep + M>1."""

        @classmethod
        def get_min_capability(cls) -> int:
            return 0

        @classmethod
        def can_implement(cls, c):
            if c.weight_type != scalar_types.uint4:
                return False, "WinRocmAwqGemv supports only uint4 (AWQ)"
            if not c.zero_points:
                return False, "WinRocmAwqGemv needs zero points (AWQ)"
            if c.has_g_idx:
                return False, "WinRocmAwqGemv does not support act-order g_idx"
            if c.group_size != 128:
                return False, "WinRocmAwqGemv supports only group_size 128"
            return True, None

        def process_weights_after_loading(self, layer) -> None:
            # Reuse conch's normalization so we consume its exact post-process layout, and keep
            # the conch instance for the M>1 (prefill/batched) path.
            from vllm.model_executor.kernels.linear.mixed_precision.conch import (
                ConchLinearKernel,
            )
            self._conch = ConchLinearKernel(
                self.config, self.w_q_name, self.w_s_name, self.w_zp_name, self.w_gidx_name
            )
            self._conch.process_weights_after_loading(layer)

        def apply_weights(self, layer, x, bias=None):
            x2d = x.reshape(-1, x.shape[-1])
            if x2d.shape[0] == 1:
                w_q, w_s, w_zp, _ = self._get_weight_params(layer)
                N = w_s.shape[1]
                out = gemv_k(x2d.contiguous(), w_q, w_s, w_zp, self.config.group_size)
                if bias is not None:
                    out = out + bias
                return out.reshape(x.shape[:-1] + (N,))
            # prefill / batched: delegate to conch (correct, same normalized weights)
            return self._conch.apply_weights(layer, x, bias)

    return WinRocmAwqGemvKernel


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
        from vllm.platforms import PlatformEnum

        kcls = _register_kernel_class()
        lst = _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, [])
        if kcls not in lst:
            lst.insert(0, kcls)
        _REGISTERED = True
        print("vllm-win: registered WinRocmAwqGemvKernel (M=1 W4 GEMV) ahead of conch")
    except Exception as e:  # noqa: BLE001
        print("vllm-win awq_gemv register warning:", repr(e))
