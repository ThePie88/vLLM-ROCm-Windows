"""Shared cpp_extension patches for torch 2.10 + ROCm on native Windows.

Imported for side effects by the sibling build_*.py scripts (one import line each). Two things in
``torch.utils.cpp_extension`` do not fit a HIP build on Windows:

  * ``_prepare_ldflags`` takes the ``IS_WINDOWS`` branch. It does pick ``c10_hip`` / ``torch_hip``
    via ``IS_HIP_EXTENSION``, but it also appends ``-INCLUDE:?warp_size@cuda@at@@YAHXZ`` -- a CUDA
    translation-unit symbol that does not exist in torch_hip, so the link fails -- and it never
    adds ``amdhip64.lib``, which a HIP extension needs. Replace the function with the HIP variant
    below. (torch 2.10 also inserted a ``with_sycl`` parameter, so the replacement accepts a
    variable arity and reads only the tail of the argument list.)
  * torch 2.10 emits the MSVC flag ``/std:c++17`` for C++ sources, which clang rejects as a
    filename: ``clang: error: no such file or directory: '/std:c++17'``. Rewrite it to
    ``-std=c++17`` in the generated ninja file.

Note on ``-fms-runtime-lib``: torch 2.10 sets ``-fms-runtime-lib=dll`` for HIP
(``COMMON_HIP_FLAGS``), i.e. the dynamic CRT. Forcing the static CRT
(``-fms-runtime-lib=static`` plus ``/DEFAULTLIB:libcmt`` ...) was tried here and measured to
BREAK the result: the .pyd compiled and linked, but importing it failed with ``WinError 1114
ERROR_DLL_INIT_FAILED``, because the torch DLLs it loads against use the dynamic CRT. This module
therefore leaves the flag alone.

Keeping this in one module means the individual build scripts stay untouched except for a single
import line.
"""
import os
import sys

from torch.utils import cpp_extension

_orig_write = cpp_extension._write_ninja_file


def _hip_prepare_ldflags(extra_ldflags, with_cuda, *args):
    # torch 2.9: (extra, with_cuda, verbose, is_standalone)
    # torch 2.10: (extra, with_cuda, with_sycl, verbose, is_standalone)
    is_standalone = args[-1] if args else False
    if os.name == "nt":
        import winrocm_paths as wp

        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")
        extra_ldflags.append("c10.lib")
        if with_cuda:
            extra_ldflags.append("c10_hip.lib")
        extra_ldflags.append("torch_cpu.lib")
        if with_cuda:
            extra_ldflags.append("torch_hip.lib")
        extra_ldflags.append("torch.lib")
        extra_ldflags.append(f"/LIBPATH:{cpp_extension.TORCH_LIB_PATH}")
        if not is_standalone:
            extra_ldflags.append("torch_python.lib")
            extra_ldflags.append(f"/LIBPATH:{python_lib_path}")
        if with_cuda:
            extra_ldflags.append(f"/LIBPATH:{wp.hip_lib()}")
            extra_ldflags.append("amdhip64.lib")
    return extra_ldflags


def _fix(flags):
    if not flags:
        return flags
    out = []
    for f in flags:
        if f == "/std:c++17":
            f = "-std=c++17"
        out.append(f)
    return out


def _write_ninja(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags,
                 cuda_dlink_post_cflags, sycl_cflags, sycl_post_cflags,
                 sycl_dlink_post_cflags, sources, objects, ldflags,
                 library_target, with_cuda, with_sycl):
    return _orig_write(path, _fix(cflags), post_cflags, _fix(cuda_cflags),
                       cuda_post_cflags, cuda_dlink_post_cflags, sycl_cflags,
                       sycl_post_cflags, sycl_dlink_post_cflags, sources, objects,
                       ldflags, library_target, with_cuda, with_sycl)


cpp_extension._prepare_ldflags = _hip_prepare_ldflags
cpp_extension._write_ninja_file = _write_ninja
