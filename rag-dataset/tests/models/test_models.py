from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.config_models import FinanceConfig, FinanceEvent, LawConfig, LawParty
from src.models.corpus_models import Document, GeneratedArticle
from src.models.query_models import (
    GeneratedQuery,
    QARPair,
    RefinedReferences,
    UnanswerableQuery,
    ValidationResult,
)


def test_finance_config_valid():
    cfg = FinanceConfig(
        company_name="Test Corp",
        industry="Banking",
        founded_year=2000,
        headquarters="Tokyo",
        events=[
            FinanceEvent(date="2023-Q1", description="Merger", financial_impact="Assets +30%"),
            FinanceEvent(date="2023-Q3", description="IPO", financial_impact="Raised 500M"),
        ],
        revenue="100B JPY",
        profit_margin="15%",
    )
    assert cfg.company_name == "Test Corp"
    assert len(cfg.events) == 2


def test_finance_config_too_few_events():
    with pytest.raises(ValidationError):
        FinanceConfig(
            company_name="X",
            industry="Banking",
            founded_year=2000,
            headquarters="Tokyo",
            events=[],  # min_length=2
            revenue="100B",
            profit_margin="10%",
        )


def test_law_config_valid():
    cfg = LawConfig(
        case_name="State v. Smith",
        court_name="High Court",
        case_date="2023-01-15",
        parties=[LawParty(name="John Smith", role="defendant")],
        charges=["Fraud"],
        key_facts=["Fact 1", "Fact 2", "Fact 3"],
        verdict="Guilty",
        sentence="2 years",
    )
    assert cfg.verdict == "Guilty"


def test_law_config_too_few_key_facts():
    with pytest.raises(ValidationError):
        LawConfig(
            case_name="X",
            court_name="Court",
            case_date="2023-01-01",
            parties=[LawParty(name="A", role="defendant")],
            charges=["Fraud"],
            key_facts=["Only one fact"],  # min_length=3
            verdict="Guilty",
            sentence="1 year",
        )


def test_document_valid():
    doc = Document(doc_id="ja_finance_0001", language="ja", domain="finance", content="Content here")
    assert doc.language == "ja"


def test_document_invalid_language():
    with pytest.raises(ValidationError):
        Document(doc_id="xx_finance_0001", language="fr", domain="finance", content="Content")


def test_qar_pair_answerable():
    q = QARPair(
        query_id="ja_finance_q0001",
        doc_id="ja_finance_0001",
        language="ja",
        domain="finance",
        query_type="factual",
        question="What is the revenue?",
        answer="100B JPY",
        references=["Revenue was 100B JPY in 2023."],
    )
    assert q.valid is True
    assert q.doc_id is not None


def test_qar_pair_unanswerable():
    q = QARPair(
        query_id="ja_finance_q0050",
        doc_id=None,
        language="ja",
        domain="finance",
        query_type="unanswerable",
        question="What is XYZ Corp's revenue?",
        answer="",
        references=[],
    )
    assert q.doc_id is None
    assert q.answer == ""


def test_generated_query_model():
    gq = GeneratedQuery(
        question="What was the merger date?",
        answer="Q1 2023",
        references=["The merger occurred in Q1 2023."],
    )
    assert len(gq.references) == 1


def test_unanswerable_query_model():
    uq = UnanswerableQuery(question="What is the population of Mars?")
    assert uq.question == "What is the population of Mars?"


def test_refined_references_model():
    rr = RefinedReferences(references=["Reference 1", "Reference 2", "Reference 3"])
    assert len(rr.references) == 3


def test_validation_result():
    vr = ValidationResult(is_valid=True, reason="Question is answerable from the document.")
    assert vr.is_valid is True
