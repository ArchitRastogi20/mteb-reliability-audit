"""Offline smoke test for S12. No GPU required for the import test."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_imports():
    mod = _load("eval_S12", REPO_ROOT / "scripts" / "eval_S12_index_build.py")
    assert hasattr(mod, "MODELS_TO_TIME")
    assert hasattr(mod, "time_one_model")


def test_models_to_time_is_nonempty():
    mod = _load("eval_S12", REPO_ROOT / "scripts" / "eval_S12_index_build.py")
    assert len(mod.MODELS_TO_TIME) >= 10
    # Spot-check that the pair from R5c is included
    ids = [m for m, _ in mod.MODELS_TO_TIME]
    assert "BAAI/bge-m3" in ids
    assert "Snowflake/snowflake-arctic-embed-l-v2.0" in ids
