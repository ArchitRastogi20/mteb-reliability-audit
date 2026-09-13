from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import PipelineConfig
from src.pipeline.orchestrator import _dry_run, main


def test_dry_run_prints_estimate(capsys):
    cfg = PipelineConfig()
    _dry_run(cfg)
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "$20.00" in out
    assert "$25.00" in out


def test_main_dry_run_flag(capsys):
    with patch("sys.argv", ["orchestrator.py", "--dry-run"]):
        main()
    out = capsys.readouterr().out
    assert "DRY RUN" in out


def test_main_requires_lang_and_domain(capsys):
    with patch("sys.argv", ["orchestrator.py"]):
        with pytest.raises(SystemExit):
            main()


def test_main_invalid_lang(capsys):
    with patch("sys.argv", ["orchestrator.py", "--lang", "fr", "--domain", "finance"]):
        with pytest.raises(SystemExit):
            main()
