from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Document(BaseModel):
    doc_id: str
    language: Literal["ja", "hi", "it"]
    domain: Literal["finance", "law"]
    content: str


class GeneratedArticle(BaseModel):
    """LLM response model for Stage 2."""
    content: str
