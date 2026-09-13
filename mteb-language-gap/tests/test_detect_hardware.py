import yaml
from pathlib import Path


def test_hardware_config_has_required_fields(tmp_path):
    from scripts.detect_hardware import detect_hardware, HardwareConfig
    cfg = detect_hardware()
    assert isinstance(cfg, HardwareConfig)
    assert cfg.vram_gb > 0
    assert cfg.ram_gb > 0
    assert cfg.cpu_cores >= 1
    assert cfg.torch_num_threads >= 1
    assert cfg.torch_num_threads <= cfg.cpu_cores


def test_hardware_config_leaves_two_threads(tmp_path):
    from scripts.detect_hardware import detect_hardware
    cfg = detect_hardware()
    assert cfg.torch_num_threads == max(1, cfg.cpu_cores - 2)


def test_batch_sizes_present(tmp_path):
    from scripts.detect_hardware import detect_hardware
    cfg = detect_hardware()
    required_tiers = {"small", "medium", "qwen_sub1b", "large", "xlarge"}
    assert required_tiers.issubset(cfg.batch_sizes.keys())
    for tier, bs in cfg.batch_sizes.items():
        assert bs >= 1, f"batch_size for {tier} must be >= 1"


def test_write_hardware_yaml(tmp_path):
    from scripts.detect_hardware import detect_hardware, write_hardware_yaml
    cfg = detect_hardware()
    out = tmp_path / "hardware.yaml"
    write_hardware_yaml(cfg, out)
    data = yaml.safe_load(out.read_text())
    assert "vram_gb" in data
    assert "batch_sizes" in data
    assert data["torch_num_threads"] == cfg.torch_num_threads


def test_vram_budget_is_95_percent(tmp_path):
    from scripts.detect_hardware import detect_hardware
    import torch
    cfg = detect_hardware()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    expected = int(total_vram * 0.95)
    assert abs(cfg.vram_budget_bytes - expected) < 1024 * 1024  # within 1MB
