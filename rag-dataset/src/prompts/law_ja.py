from __future__ import annotations

from pydantic import BaseModel

from src.prompts import PromptSet


class LawJaPrompts(PromptSet):
    language = "ja"
    domain = "law"

    @property
    def config_system(self) -> str:
        return "あなたは日本の法律判例のデータを生成する専門家です。指定された形式で正確に判例設定データを生成してください。"

    @property
    def article_system(self) -> str:
        return "あなたは日本語で判決文を執筆する専門家です。提供された判例設定に基づいて詳細な判決文を作成してください。"

    @property
    def query_system(self) -> str:
        return "あなたはRAG評価用の法律質問を生成する専門家です。指示に従って正確な形式で質問を生成してください。"

    @property
    def refine_system(self) -> str:
        return "あなたは判決文から正確な引用文を抽出する専門家です。引用文は判決文に完全に一致している必要があります。"

    def config_user(self, index: int) -> str:
        return (
            f"日本の法律判例の設定データを生成してください（事件番号: {index}）。"
            "刑事事件（詐欺、横領、傷害、窃盗など）、民事事件（契約違反、損害賠償）、"
            "行政訴訟などから選んでください。"
            "事件名、裁判所名、判決日、当事者（被告・検察官など）、罪状、"
            "主要な事実（3〜5件）、判決、量刑をすべて日本語で記述してください。"
        )

    def article_user(self, config: BaseModel) -> str:
        config_json = config.model_dump_json(indent=2)
        return (
            "以下の判例設定に基づいて、800〜1200語の詳細な日本語判決文を作成してください。\n"
            "事件の概要、犯罪事実の詳細な説明、証拠の概要、法的判断、量刑理由を含めてください。\n"
            "すべて日本語で記述し、英語は使用しないでください。\n\n"
            f"設定:\n{config_json}"
        )

    def factual_query_user(self, article: str) -> str:
        return (
            "以下の日本語の判決文から、1つの事実的な質問を生成してください。\n"
            "質問は判決文の1文から答えられるものにしてください。\n"
            "質問、期待される回答、および回答を支持する正確な引用文を1〜2つ含めてください。\n"
            "すべて日本語で記述してください。\n\n"
            f"判決文:\n{article}"
        )

    def multihop_query_user(self, article: str) -> str:
        return (
            "以下の日本語の判決文から、複合推論質問を1つ生成してください。\n"
            "質問は判決文内の2つ以上の異なる情報を組み合わせて答える必要があるものにしてください。\n"
            "質問、詳細な回答、および2つ以上の正確な引用文を含めてください。\n"
            "すべて日本語で記述してください。\n\n"
            f"判決文:\n{article}"
        )

    def summarization_query_user(self, article: str) -> str:
        return (
            "以下の日本語の判決文全体を読んで、要約質問を1つ生成してください。\n"
            "質問は判決文全体の理解が必要な要約を求めるものにしてください。\n"
            "質問、包括的な回答、および主要な引用文を2〜3つ含めてください。\n"
            "すべて日本語で記述してください。\n\n"
            f"判決文:\n{article}"
        )

    def unanswerable_query_user(self, existing_entities: list[str]) -> str:
        entities_str = "、".join(existing_entities[:15]) if existing_entities else "なし"
        return (
            "日本の法律・裁判に関する質問を1つ生成してください。\n"
            "ただし、以下のリストに含まれない架空の人物または事件についての質問にしてください。\n"
            "質問のみを日本語で生成してください（回答は不要です）。\n\n"
            f"既存の事件・人物（これらとは異なるものを使うこと）: {entities_str}"
        )

    def refine_user(self, question: str, answer: str, article: str) -> str:
        return (
            "以下の質問と回答を支持する、判決文からの正確な引用文を抽出してください。\n"
            "引用文は判決文に完全に一致する文字列である必要があります。\n"
            "最も関連性の高い1〜3つの引用文を抽出してください。\n\n"
            f"質問: {question}\n回答: {answer}\n\n判決文:\n{article}"
        )
