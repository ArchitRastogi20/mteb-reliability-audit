from __future__ import annotations

import json

from pydantic import BaseModel

from src.prompts import PromptSet


class FinanceJaPrompts(PromptSet):
    language = "ja"
    domain = "finance"

    def config_user(self, index: int) -> str:
        return (
            f"日本の金融企業の設定データを生成してください（企業番号: {index}）。"
            "銀行、証券、保険、不動産、商社、リース、信託などの業種から選んでください。"
            "企業名、業種、設立年、本社所在地、2〜3つの重要なイベント（日付・説明・財務的影響）、"
            "収益、利益率をすべて日本語で記述してください。"
        )

    def article_user(self, config: BaseModel) -> str:
        config_json = config.model_dump_json(indent=2)
        return (
            f"以下の企業設定に基づいて、800〜1200語の詳細な日本語年次報告書を作成してください。\n"
            "企業概要、財務実績、重要イベントの詳細な説明、市場分析、今後の展望を含めてください。"
            "すべて日本語で記述し、英語は使用しないでください。\n\n"
            f"設定:\n{config_json}"
        )

    def factual_query_user(self, article: str) -> str:
        return (
            "以下の日本語の記事から、1つの事実的な質問を生成してください。\n"
            "質問は記事の1文から答えられるものにしてください。\n"
            "質問、期待される回答、および回答を支持する記事からの正確な引用文を1〜2つ含めてください。\n"
            "すべて日本語で記述してください。\n\n"
            f"記事:\n{article}"
        )

    def multihop_query_user(self, article: str) -> str:
        return (
            "以下の日本語の記事から、複合推論質問を1つ生成してください。\n"
            "質問は記事内の2つ以上の異なる情報を組み合わせて答える必要があるものにしてください。\n"
            "質問、詳細な回答、および2つ以上の正確な引用文を含めてください。\n"
            "すべて日本語で記述してください。\n\n"
            f"記事:\n{article}"
        )

    def summarization_query_user(self, article: str) -> str:
        return (
            "以下の日本語の記事全体を読んで、要約質問を1つ生成してください。\n"
            "質問は記事全体の理解が必要な要約を求めるものにしてください。\n"
            "質問、包括的な回答、および主要な引用文を2〜3つ含めてください。\n"
            "すべて日本語で記述してください。\n\n"
            f"記事:\n{article}"
        )

    def unanswerable_query_user(self, existing_entities: list[str]) -> str:
        entities_str = "、".join(existing_entities[:15]) if existing_entities else "なし"
        return (
            "日本の金融分野に関する質問を1つ生成してください。\n"
            "ただし、以下のリストに含まれない架空の企業についての質問にしてください。\n"
            "質問は一見もっともらしく、実在する企業に関する質問のように見えるものにしてください。\n"
            "質問のみを日本語で生成してください（回答は不要です）。\n\n"
            f"既存の企業（これらとは異なる企業を使うこと）: {entities_str}"
        )

    def refine_user(self, question: str, answer: str, article: str) -> str:
        return (
            "以下の質問と回答を支持する、記事からの正確な引用文を抽出してください。\n"
            "引用文は記事に完全に一致する文字列である必要があります。\n"
            "最も関連性の高い1〜3つの引用文を抽出してください。\n\n"
            f"質問: {question}\n"
            f"回答: {answer}\n\n"
            f"記事:\n{article}"
        )

    @property
    def config_system(self) -> str:
        return "あなたは日本の金融企業のデータを生成する専門家です。指定された形式で正確に企業設定データを生成してください。"

    @property
    def article_system(self) -> str:
        return "あなたは日本語で金融レポートを執筆する専門家です。提供された企業設定に基づいて詳細な年次報告書を作成してください。"

    @property
    def query_system(self) -> str:
        return "あなたはRAG評価用の質問を生成する専門家です。指示に従って正確な形式で質問を生成してください。"

    @property
    def refine_system(self) -> str:
        return "あなたは文書から正確な引用文を抽出する専門家です。引用文は文書に完全に一致している必要があります。"
