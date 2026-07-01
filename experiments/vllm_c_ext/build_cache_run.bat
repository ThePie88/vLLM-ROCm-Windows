@echo off
call "E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROCM_HOME=C:\HIP-SDK
set HIP_PATH=C:\HIP-SDK
set ROCM_PATH=C:\HIP-SDK
if exist C:\vw_cache_build rmdir /s /q C:\vw_cache_build
if exist C:\vw_cache_hip rmdir /s /q C:\vw_cache_hip
python -u "C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\experiments\vllm_c_ext\build_cache_c.py"
