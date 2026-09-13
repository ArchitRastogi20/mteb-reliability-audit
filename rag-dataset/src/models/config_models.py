from __future__ import annotations

from pydantic import BaseModel, Field


class FinanceEvent(BaseModel):
    date: str
    description: str
    financial_impact: str


class FinanceConfig(BaseModel):
    company_name: str
    industry: str
    founded_year: int
    headquarters: str
    events: list[FinanceEvent] = Field(min_length=2, max_length=3)
    revenue: str
    profit_margin: str


class LawParty(BaseModel):
    name: str
    role: str


class LawConfig(BaseModel):
    case_name: str
    court_name: str
    case_date: str
    parties: list[LawParty] = Field(min_length=1)
    charges: list[str] = Field(min_length=1)
    key_facts: list[str] = Field(min_length=3, max_length=5)
    verdict: str
    sentence: str
