"""Local inspection only; no model downloads or web access."""
from __future__ import annotations

import ctypes.util
import importlib.metadata
import importlib.util
import os
import platform
import shutil


def environment_report() -> dict:
    packages = {}
    for name in ("pygame-ce", "sounddevice", "soundfile", "numpy", "librosa", "pytest", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"os": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "display_available": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or platform.system() != "Linux"),
              "packages": packages, "portaudio_library": ctypes.util.find_library("portaudio"),
              "gpu": {"pytorch_installed": packages["torch"] is not None,
                      "nvidia_smi": shutil.which("nvidia-smi"), "cuda": None, "mps": None},
              "audio": {"devices": [], "output_devices": [], "error": None}}
    if importlib.util.find_spec("torch"):
        try:
            import torch
            report["gpu"].update(torch_build=str(torch.__version__), compiled_cuda=torch.version.cuda,
                                 cuda=torch.cuda.is_available(),
                                 mps=hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception as error:
            report["gpu"]["error"] = str(error)
    else:
        report["gpu"]["note"] = "PyTorch is not installed; CUDA/MPS inference was not tested or required."
    try:
        from .audio_engine import get_sounddevice
        sd = get_sounddevice()
        devices = [dict(device, index=i) for i, device in enumerate(sd.query_devices())]
        report["audio"].update(devices=devices,
                                output_devices=[d for d in devices if d["max_output_channels"] > 0],
                                host_apis=list(sd.query_hostapis()),
                                default_device=list(sd.default.device),
                                portaudio_version=list(sd.get_portaudio_version()))
    except Exception as error:
        report["audio"]["error"] = str(error)
    return report
