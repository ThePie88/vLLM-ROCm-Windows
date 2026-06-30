@echo off
call "E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROCM_HOME=C:\HIP-SDK
set HIP_PATH=C:\HIP-SDK
set ROCM_PATH=C:\HIP-SDK
set HF_HUB_OFFLINE=1
if exist C:\vw_hipgemv_build rmdir /s /q C:\vw_hipgemv_build
set TORCH_EXTENSIONS_DIR=C:\vw_hipgemv_build
python -u "C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\experiments\w4_gemv\hip\build_test_hip.py"
