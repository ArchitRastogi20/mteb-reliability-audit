from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import torch
import yaml

REPO_ROOT = Path(__file__).parent.parent

VRAM_TIERS: list[tuple[float, dict[str, int]]] = [
    (80.0, {"small": 4096, "medium": 2048, "qwen_sub1b": 512,  "large": 256, "xlarge": 128}),
    (40.0, {"small": 2048, "medium": 1024, "qwen_sub1b": 256,  "large": 128, "xlarge": 64}),
    (24.0, {"small": 1024, "medium": 512,  "qwen_sub1b": 128,  "large": 64,  "xlarge": 32}),
    (0.0,  {"small": 512,  "medium": 256,  "qwen_sub1b": 64,   "large": 32,  "xlarge": 16}),
]


@dataclass
class HardwareConfig:
    vram_gb: float
    ram_gb: float
    cpu_cores: int
    torch_num_threads: int
    vram_budget_bytes: int
    ram_budget_bytes: int
    batch_sizes: dict[str, int] = field(default_factory=dict)


def detect_hardware() -> HardwareConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for evaluation")

    total_vram = torch.cuda.get_device_properties(0).total_memory
    vram_gb = total_vram / 1e9

    ram_info = psutil.virtual_memory()
    ram_gb = ram_info.total / 1e9

    cpu_cores = os.cpu_count() or 4
    torch_num_threads = max(1, cpu_cores - 2)

    vram_budget_bytes = int(total_vram * 0.95)
    ram_budget_bytes = int(ram_info.total * 0.95)

    batch_sizes: dict[str, int] = {}
    for threshold_gb, sizes in VRAM_TIERS:
        if vram_gb >= threshold_gb:
            batch_sizes = sizes
            break

    return HardwareConfig(
        vram_gb=round(vram_gb, 1),
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        torch_num_threads=torch_num_threads,
        vram_budget_bytes=vram_budget_bytes,
        ram_budget_bytes=ram_budget_bytes,
        batch_sizes=batch_sizes,
    )


def write_hardware_yaml(
    cfg: HardwareConfig,
    path: Path = REPO_ROOT / "config" / "hardware.yaml",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({
        "vram_gb": cfg.vram_gb,
        "ram_gb": cfg.ram_gb,
        "cpu_cores": cfg.cpu_cores,
        "torch_num_threads": cfg.torch_num_threads,
        "vram_budget_bytes": cfg.vram_budget_bytes,
        "ram_budget_bytes": cfg.ram_budget_bytes,
        "batch_sizes": cfg.batch_sizes,
    }, default_flow_style=False))


def detect_and_write(
    path: Path = REPO_ROOT / "config" / "hardware.yaml",
) -> HardwareConfig:
    cfg = detect_hardware()
    write_hardware_yaml(cfg, path)
    return cfg
