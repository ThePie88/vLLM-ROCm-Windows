"""Build vLLM's csrc fused-op kernels (activation/layernorm/pos_encoding) into a native
Windows `torch.ops._C` extension — 1:1 from vLLM's own ROCm sources via the Gate-A recipe.

Run with MSVC env + ROCM_HOME=C:\\HIP-SDK (see experiments/phase0_hip_ext/README.md).
"""
import os
import sys
import time

import torch
from torch.utils import cpp_extension

# Gate-A hipify None-guard.
from torch.utils.hipify import hipify_python as _hp
_orig = _hp.hipify
def _no_none(*a, **k):
    r = _orig(*a, **k)
    try:
        for key, v in r.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception:
        pass
    return r
_hp.hipify = _no_none

VLLM_CSRC = r"C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\vllm\csrc"
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = r"C:\vw_cext_build"
DEVICE_LIB = r"C:\HIP-SDK\lib\llvm\amdgcn\bitcode"
os.makedirs(BUILD_DIR, exist_ok=True)

sources = [
    os.path.join(VLLM_CSRC, "activation_kernels.cu"),
    os.path.join(VLLM_CSRC, "layernorm_kernels.cu"),
    os.path.join(VLLM_CSRC, "pos_encoding_kernels.cu"),
    os.path.join(HERE, "win_c_bindings.cpp"),
]
print("torch", torch.__version__, "hip", torch.version.hip)
print("sources:", [os.path.basename(s) for s in sources])
sys.stdout.flush()

t0 = time.perf_counter()
ext = cpp_extension.load(
    name="vllm_win_C",
    sources=sources,
    extra_include_paths=[VLLM_CSRC],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
        f"-I{VLLM_CSRC}",
    ],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")

# correctness vs torch
import torch.nn.functional as F
x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
out = torch.empty(64, 2048, device="cuda", dtype=torch.float16)
torch.ops._C.silu_and_mul(out, x)
ref = F.silu(x[..., :2048]) * x[..., 2048:]
print("SILU_OK", bool((out - ref).abs().max().item() < 1e-2))

w = torch.randn(4096, device="cuda", dtype=torch.float16)
rn = torch.empty_like(x)
torch.ops._C.rms_norm(rn, x, w, 1e-6)
xr = x.float(); refn = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16) * w
print("RMSNORM_OK", bool((rn - refn).abs().max().item() < 2e-2))
print("CEXT_OK")
