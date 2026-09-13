import yaml


def test_hardware_config_fields():
    from scripts.rag_hardware import detect_hardware, HardwareConfig
    cfg = detect_hardware()
    assert isinstance(cfg, HardwareConfig)
    assert cfg.vram_gb > 0
    assert cfg.ram_gb > 0
    assert cfg.cpu_cores >= 1
    assert cfg.torch_num_threads >= 1
    assert cfg.torch_num_threads <= cfg.cpu_cores


def test_torch_threads_leaves_two_cores():
    from scripts.rag_hardware import detect_hardware
    cfg = detect_hardware()
    assert cfg.torch_num_threads == max(1, cfg.cpu_cores - 2)


def test_batch_sizes_all_tiers_present():
    from scripts.rag_hardware import detect_hardware
    cfg = detect_hardware()
    required = {"small", "medium", "qwen_sub1b", "large", "xlarge"}
    assert required.issubset(cfg.batch_sizes.keys())
    for tier, bs in cfg.batch_sizes.items():
        assert bs >= 1


def test_vram_budget_is_95_percent():
    from scripts.rag_hardware import detect_hardware
    import torch
    cfg = detect_hardware()
    total = torch.cuda.get_device_properties(0).total_memory
    assert abs(cfg.vram_budget_bytes - int(total * 0.95)) < 1024 * 1024


def test_ram_budget_is_95_percent():
    from scripts.rag_hardware import detect_hardware
    import psutil
    cfg = detect_hardware()
    total = psutil.virtual_memory().total
    assert abs(cfg.ram_budget_bytes - int(total * 0.95)) < 1024 * 1024


def test_write_hardware_yaml(tmp_path):
    from scripts.rag_hardware import detect_hardware, write_hardware_yaml
    cfg = detect_hardware()
    out = tmp_path / "hardware.yaml"
    write_hardware_yaml(cfg, out)
    data = yaml.safe_load(out.read_text())
    assert "vram_gb" in data
    assert "ram_gb" in data
    assert "batch_sizes" in data
    assert data["torch_num_threads"] == cfg.torch_num_threads
