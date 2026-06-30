@echo off
call "E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set ROCM_HOME=C:\HIP-SDK
set HIP_PATH=C:\HIP-SDK
set ROCM_PATH=C:\HIP-SDK
if exist C:\vw_cext_build rmdir /s /q C:\vw_cext_build
if exist C:\vw_cext_hip rmdir /s /q C:\vw_cext_hip
python -u "C:\Users\filip\Desktop\Progetto_VLLM_ROCM_WINDOWS\experiments\vllm_c_ext\build_c_ext.py"
