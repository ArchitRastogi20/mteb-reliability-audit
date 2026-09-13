from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.config import PipelineConfig


def test_defaults():
    cfg = PipelineConfig.load(path="nonexistent_file.yaml")
    assert cfg.corpus.docs_per_combo == 250
    assert cfg.queries.factual == 175
    assert cfg.queries.total == 500
    assert cfg.budget_usd == 25.0
    assert cfg.models.config_gen == "gpt-5.4-nano"


def test_load_from_yaml(tmp_path):
    config_data = {
        "corpus": {"docs_per_combo": 50},
        "queries": {"factual": 20, "multi_hop": 20, "summarization": 5, "unanswerable": 5},
        "budget_usd": 10.0,
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")

    cfg = PipelineConfig.load(config_file)
    assert cfg.corpus.docs_per_combo == 50
    assert cfg.queries.factual == 20
    assert cfg.queries.total == 50
    assert cfg.budget_usd == 10.0
    # Unspecified values use defaults
    assert cfg.models.config_gen == "gpt-5.4-nano"
    assert cfg.semaphores.stage1 == 50


def test_query_total():
    cfg = PipelineConfig()
    cfg.queries.factual = 10
    cfg.queries.multi_hop = 10
    cfg.queries.summarization = 5
    cfg.queries.unanswerable = 5
    assert cfg.queries.total == 30


def test_dry_run_uses_config(capsys):
    from src.pipeline.orchestrator import _dry_run
    cfg = PipelineConfig()
    cfg.corpus.docs_per_combo = 100
    cfg.queries.factual = 50
    cfg.queries.multi_hop = 50
    cfg.queries.summarization = 20
    cfg.queries.unanswerable = 10
    _dry_run(cfg)
    out = capsys.readouterr().out
    assert "100" in out
    assert "$25.00" in out
