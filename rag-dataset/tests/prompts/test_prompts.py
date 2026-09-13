from __future__ import annotations

from src.models.config_models import FinanceConfig, FinanceEvent, LawConfig, LawParty
from src.prompts.finance_hi import FinanceHiPrompts
from src.prompts.finance_it import FinanceItPrompts
from src.prompts.finance_ja import FinanceJaPrompts
from src.prompts.law_hi import LawHiPrompts
from src.prompts.law_it import LawItPrompts
from src.prompts.law_ja import LawJaPrompts


def _finance_config() -> FinanceConfig:
    return FinanceConfig(
        company_name="TestCorp",
        industry="Banking",
        founded_year=2000,
        headquarters="Tokyo",
        events=[
            FinanceEvent(date="2023-Q1", description="Merger", financial_impact="+30%"),
            FinanceEvent(date="2023-Q3", description="IPO", financial_impact="500M"),
        ],
        revenue="100B",
        profit_margin="15%",
    )


def _law_config() -> LawConfig:
    return LawConfig(
        case_name="State v. Doe",
        court_name="High Court",
        case_date="2023-01-15",
        parties=[LawParty(name="John Doe", role="defendant")],
        charges=["Fraud"],
        key_facts=["Fact 1", "Fact 2", "Fact 3"],
        verdict="Guilty",
        sentence="2 years",
    )


def test_finance_ja_config_user_has_index():
    assert "42" in FinanceJaPrompts().config_user(42)

def test_finance_ja_article_user_has_config_name():
    assert "TestCorp" in FinanceJaPrompts().article_user(_finance_config())

def test_finance_ja_factual_has_article():
    assert "article text" in FinanceJaPrompts().factual_query_user("article text")

def test_finance_ja_multihop_has_article():
    assert "article text" in FinanceJaPrompts().multihop_query_user("article text")

def test_finance_ja_summarization_has_article():
    assert "article text" in FinanceJaPrompts().summarization_query_user("article text")

def test_finance_ja_unanswerable_lists_entities():
    assert "CorpA" in FinanceJaPrompts().unanswerable_query_user(["CorpA", "CorpB"])

def test_finance_ja_refine_has_question_and_article():
    p = FinanceJaPrompts().refine_user("Q?", "A.", "article text")
    assert "Q?" in p and "article text" in p

def test_finance_hi_config_user_has_index():
    assert "7" in FinanceHiPrompts().config_user(7)

def test_finance_hi_article_user_has_config_name():
    assert "TestCorp" in FinanceHiPrompts().article_user(_finance_config())

def test_law_ja_config_user_has_index():
    assert "10" in LawJaPrompts().config_user(10)

def test_law_ja_article_user_has_case_name():
    assert "State v. Doe" in LawJaPrompts().article_user(_law_config())

def test_law_hi_config_user_has_index():
    assert "3" in LawHiPrompts().config_user(3)

def test_law_hi_article_user_has_case_name():
    assert "State v. Doe" in LawHiPrompts().article_user(_law_config())

def test_all_system_prompts_non_empty():
    for cls in [FinanceJaPrompts, FinanceHiPrompts, LawJaPrompts, LawHiPrompts]:
        p = cls()
        assert p.config_system and p.article_system and p.query_system and p.refine_system


def _law_config_it() -> LawConfig:
    return LawConfig(
        case_name="Stato c. Rossi",
        court_name="Tribunale di Milano",
        case_date="2023-03-15",
        parties=[LawParty(name="Marco Rossi", role="defendant")],
        charges=["Frode fiscale"],
        key_facts=["Fatto 1", "Fatto 2", "Fatto 3"],
        verdict="Colpevole",
        sentence="2 anni",
    )


def test_finance_it_config_user_has_index():
    assert "5" in FinanceItPrompts().config_user(5)

def test_finance_it_article_user_has_config():
    assert "TestCorp" in FinanceItPrompts().article_user(_finance_config())

def test_finance_it_factual_has_article():
    assert "testo" in FinanceItPrompts().factual_query_user("testo articolo")

def test_finance_it_unanswerable_lists_entities():
    assert "Alfa" in FinanceItPrompts().unanswerable_query_user(["Alfa", "Beta"])

def test_law_it_config_user_has_index():
    assert "8" in LawItPrompts().config_user(8)

def test_law_it_article_user_has_case():
    assert "Stato c. Rossi" in LawItPrompts().article_user(_law_config_it())

def test_all_it_system_prompts_non_empty():
    for cls in [FinanceItPrompts, LawItPrompts]:
        p = cls()
        assert p.config_system and p.article_system and p.query_system and p.refine_system
