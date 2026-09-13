#!/usr/bin/env python3
"""Detect and report available hardware for deployment benchmarking."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field

import psutil
import torch

# VRAM thresholds (GB) → batch sizes per model tier
_VRAM_TIERS: list[tuple[float, dict[str, int]]] = [
    (80.0, {"small": 4096, "medium": 2048, "large": 256, "xlarge": 128}),
    (40.0, {"small": 2048, "medium": 1024, "large": 128, "xlarge": 64}),
    (24.0, {"small": 1024, "medium": 512,  "large": 64,  "xlarge": 32}),
    (0.0,  {"small": 512,  "medium": 256,  "large": 32,  "xlarge": 16}),
]


@dataclass
class HardwareInfo:
    gpu_name: str
    vram_gb: float
    vram_total_bytes: int
    ram_gb: float
    cpu_cores: int
    cpu_model: str
    torch_version: str
    cuda_version: str
    platform: str
    batch_sizes: dict[str, int] = field(default_factory=dict)


def detect() -> HardwareInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected — a CUDA GPU is required")

    props = torch.cuda.get_device_properties(0)
    vram_bytes = props.total_memory
    vram_gb = vram_bytes / 1e9

    ram_gb = psutil.virtual_memory().total / 1e9
    cpu_cores = os.cpu_count() or 1

    try:
        cpu_model = platform.processor() or "unknown"
    except Exception:
        cpu_model = "unknown"

    batch_sizes: dict[str, int] = {}
    for threshold, sizes in _VRAM_TIERS:
        if vram_gb >= threshold:
            batch_sizes = dict(sizes)
            break

    return HardwareInfo(
        gpu_name=props.name,
        vram_gb=round(vram_gb, 1),
        vram_total_bytes=vram_bytes,
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        cpu_model=cpu_model,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "unknown",
        platform=platform.platform(),
        batch_sizes=batch_sizes,
    )


def print_summary(hw: HardwareInfo) -> None:
    print("=" * 60)
    print("Hardware Summary")
    print("=" * 60)
    print(f"GPU:          {hw.gpu_name}")
    print(f"VRAM:         {hw.vram_gb:.1f} GB")
    print(f"RAM:          {hw.ram_gb:.1f} GB")
    print(f"CPU cores:    {hw.cpu_cores}")
    print(f"Torch:        {hw.torch_version}")
    print(f"CUDA:         {hw.cuda_version}")
    print(f"Platform:     {hw.platform}")
    print(f"Batch sizes:  {hw.batch_sizes}")
    print("=" * 60)


if __name__ == "__main__":
    hw = detect()
    print_summary(hw)
