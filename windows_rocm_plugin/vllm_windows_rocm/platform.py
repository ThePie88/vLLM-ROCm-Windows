"""WindowsRocmPlatform: vLLM RocmPlatform adapted for native Windows.

RocmPlatform itself imports cleanly on Windows (its `amdsmi` import is wrapped in
try/except) and most methods already work: get_device_capability() derives (major, minor)
from the module-load gcnArchName (resolved via torch.cuda fallback on Windows), and
get_device_total_memory()/set_device() use torch.cuda. We only override the handful of
methods that hard-depend on amdsmi, routing them through torch.cuda instead.
"""
import torch

from vllm.platforms.rocm import RocmPlatform


class WindowsRocmPlatform(RocmPlatform):
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        try:
            uuid = getattr(torch.cuda.get_device_properties(device_id), "uuid", None)
            return str(uuid) if uuid is not None else f"gpu-{device_id}"
        except Exception:
            return f"gpu-{device_id}"

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        # Single-GPU only on Windows (no amdsmi topology, no RCCL). Never "fully connected".
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        super().check_and_update_config(vllm_config)
        # Register the fast M=1 AWQ-uint4 GEMV ahead of conch. Done here (engine config setup)
        # rather than at import/bootstrap: vllm + the platform plugin are fully loaded by now,
        # so importing vllm.model_executor.kernels.linear won't circular-import this package.
        try:
            from . import awq_gemv
            awq_gemv.register()
        except Exception as e:  # noqa: BLE001
            print("vllm-win awq_gemv register warning:", repr(e))
