from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class PromptSet(ABC):
    """Abstract base defining all prompts for one language+domain combination."""

    language: str
    domain: str

    @abstractmethod
    def config_user(self, index: int) -> str: ...

    @abstractmethod
    def article_user(self, config: BaseModel) -> str: ...

    @abstractmethod
    def factual_query_user(self, article: str) -> str: ...

    @abstractmethod
    def multihop_query_user(self, article: str) -> str: ...

    @abstractmethod
    def summarization_query_user(self, article: str) -> str: ...

    @abstractmethod
    def unanswerable_query_user(self, existing_entities: list[str]) -> str: ...

    @abstractmethod
    def refine_user(self, question: str, answer: str, article: str) -> str: ...
