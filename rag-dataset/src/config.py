from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CorpusConfig(BaseModel):
    docs_per_combo: int = 250


class QueryConfig(BaseModel):
    factual: int = 175
    multi_hop: int = 175
    summarization: int = 100
    unanswerable: int = 50

    @property
    def total(self) -> int:
        return self.factual + self.multi_hop + self.summarization + self.unanswerable


class ModelConfig(BaseModel):
    config_gen: str = "gpt-5.4-nano"
    article_gen: str = "gpt-5.4-mini"
    factual_query: str = "gpt-5.4-nano"
    complex_query: str = "gpt-5.4-mini"
    refinement: str = "gpt-5.4-nano"
    validation: str = "gpt-5.4-nano"
    answer_gen: str = "gpt-5.4-mini"
    keypoint_gen: str = "gpt-5.4-nano"
    answer_validation: str = "gpt-5.4-nano"


class SemaphoreConfig(BaseModel):
    stage1: int = 50
    stage2: int = 30
    stage3: int = 40
    stage4: int = 50
    stage5: int = 30
    validation: int = 50


class PathConfig(BaseModel):
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "output"
    evaluation_dir: str = "evaluation"


class PipelineConfig(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    queries: QueryConfig = Field(default_factory=QueryConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    semaphores: SemaphoreConfig = Field(default_factory=SemaphoreConfig)
    budget_usd: float = 25.0
    paths: PathConfig = Field(default_factory=PathConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PipelineConfig":
        """Load from YAML file, falling back to defaults if file not found."""
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config.yaml"
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
