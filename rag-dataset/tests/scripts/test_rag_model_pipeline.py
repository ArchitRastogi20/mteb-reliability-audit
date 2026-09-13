from pathlib import Path
from unittest.mock import patch


def test_pipeline_yields_all_models(tmp_path):
    from scripts.rag_model_pipeline import ModelPipeline

    def fake_download(repo_id, cache_dir, local_files_only=False):
        p = Path(cache_dir) / repo_id.replace("/", "__")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    model_ids = ["org/model-a", "org/model-b", "org/model-c"]
    yielded = []
    with patch("scripts.rag_model_pipeline.snapshot_download", side_effect=fake_download):
        pipeline = ModelPipeline(model_ids, cache_dir=tmp_path / "cache", disk_budget=int(1e12))
        for mid, path in pipeline:
            yielded.append(mid)

    assert yielded == model_ids


def test_pipeline_cleans_up_model_dir(tmp_path):
    from scripts.rag_model_pipeline import ModelPipeline

    model_ids = ["org/model-x"]

    def fake_download(repo_id, cache_dir, local_files_only=False):
        p = Path(cache_dir) / repo_id.replace("/", "__")
        p.mkdir(parents=True, exist_ok=True)
        (p / "weights.bin").write_bytes(b"fake")
        return str(p)

    yielded_paths = []
    with patch("scripts.rag_model_pipeline.snapshot_download", side_effect=fake_download):
        pipeline = ModelPipeline(model_ids, cache_dir=tmp_path / "cache", disk_budget=int(1e12))
        for mid, path in pipeline:
            yielded_paths.append(path)

    for p in yielded_paths:
        assert not Path(p).exists(), f"{p} should have been deleted after use"


def test_pipeline_yields_model_id_and_path(tmp_path):
    from scripts.rag_model_pipeline import ModelPipeline

    def fake_download(repo_id, cache_dir, local_files_only=False):
        p = Path(cache_dir) / repo_id.replace("/", "__")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    with patch("scripts.rag_model_pipeline.snapshot_download", side_effect=fake_download):
        pipeline = ModelPipeline(["org/m1"], cache_dir=tmp_path / "cache", disk_budget=int(1e12))
        for mid, path in pipeline:
            assert mid == "org/m1"
            assert Path(path).exists()


def test_pipeline_raises_on_download_failure(tmp_path):
    from scripts.rag_model_pipeline import ModelPipeline
    import pytest

    with patch("scripts.rag_model_pipeline.snapshot_download", side_effect=RuntimeError("network error")):
        pipeline = ModelPipeline(["org/m1"], cache_dir=tmp_path / "cache", disk_budget=int(1e12))
        with pytest.raises(RuntimeError, match="network error"):
            list(pipeline)
