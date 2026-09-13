import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_disk_quota_calculation():
    from scripts.utils import DISK_BUDGET_BYTES, disk_usage_bytes
    total = shutil.disk_usage("/workspace").total
    assert DISK_BUDGET_BYTES == int(total * 0.70)


def test_model_pipeline_iterates_all(tmp_path):
    """ModelPipeline should yield every model_id exactly once."""
    from scripts.model_pipeline import ModelPipeline

    downloaded = []
    def fake_download(repo_id, cache_dir, local_files_only=False):
        p = tmp_path / "cache" / repo_id.replace("/", "__")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    model_ids = ["org/model-a", "org/model-b", "org/model-c"]
    with patch("scripts.model_pipeline.snapshot_download", side_effect=fake_download):
        pipeline = ModelPipeline(model_ids, cache_dir=tmp_path / "cache", disk_budget=int(1e12))
        for mid, path in pipeline:
            downloaded.append(mid)

    assert downloaded == model_ids


def test_cleanup_removes_model_dir(tmp_path):
    """After iterating, each model's directory should be deleted."""
    from scripts.model_pipeline import ModelPipeline

    model_ids = ["org/model-x"]

    def fake_download(repo_id, cache_dir, local_files_only=False):
        p = tmp_path / "cache" / repo_id.replace("/", "__")
        p.mkdir(parents=True, exist_ok=True)
        (p / "model.bin").write_bytes(b"fake")
        return str(p)

    with patch("scripts.model_pipeline.snapshot_download", side_effect=fake_download):
        pipeline = ModelPipeline(model_ids, cache_dir=tmp_path / "cache", disk_budget=int(1e12))
        yielded_paths = []
        for mid, path in pipeline:
            yielded_paths.append(path)

    for p in yielded_paths:
        assert not Path(p).exists(), f"{p} should have been deleted"
