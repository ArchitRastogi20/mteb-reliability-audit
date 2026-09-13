from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class QARPair(BaseModel):
    query_id: str
    doc_id: str | None  # None for unanswerable queries
    language: Literal["ja", "hi", "it"]
    domain: Literal["finance", "law"]
    query_type: Literal["factual", "multi_hop", "summarization", "unanswerable"]
    question: str
    answer: str  # empty string for unanswerable
    references: list[str]  # empty for unanswerable
    keypoints: list[str] = []  # key facts extracted from answer (RAGEval-style)
    generated_answer: str = ""  # LLM answer generated from GT references
    valid: bool = True
    answer_valid: bool = True  # False if generated_answer fails coverage check


class GeneratedQuery(BaseModel):
    """LLM response model for factual / multi-hop / summarization queries."""
    question: str
    answer: str
    references: list[str]


class UnanswerableQuery(BaseModel):
    """LLM response model for unanswerable queries."""
    question: str


class RefinedReferences(BaseModel):
    """LLM response model for Stage 4 reference refinement."""
    references: list[str]


class ValidationResult(BaseModel):
    """LLM response model for LLM validator."""
    is_valid: bool
    reason: str


class RagAnswer(BaseModel):
    """LLM response model for Stage 5 RAG answer generation."""
    content: str


class KeypointExtraction(BaseModel):
    """LLM response model for keypoint extraction from expected answer."""
    keypoints: list[str]


class AnswerValidation(BaseModel):
    """LLM response model for answer quality validation."""
    is_valid: bool
    score: float  # 0.0–1.0 coverage of expected keypoints
    reason: str
