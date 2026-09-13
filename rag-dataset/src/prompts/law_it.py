from __future__ import annotations

from pydantic import BaseModel

from src.prompts import PromptSet


class LawItPrompts(PromptSet):
    language = "it"
    domain = "law"

    @property
    def config_system(self) -> str:
        return "Sei un esperto nella generazione di dati su casi giuridici italiani. Genera dati di configurazione del caso nel formato specificato."

    @property
    def article_system(self) -> str:
        return "Sei un esperto nella redazione di sentenze giudiziarie in italiano. Sulla base della configurazione del caso fornita, crea una sentenza dettagliata interamente in italiano."

    @property
    def query_system(self) -> str:
        return "Sei un esperto nella generazione di domande giuridiche per la valutazione RAG. Genera domande nel formato esatto richiesto."

    @property
    def refine_system(self) -> str:
        return "Sei un esperto nell'estrazione di citazioni esatte da sentenze. Le citazioni devono corrispondere esattamente al testo della sentenza."

    def config_user(self, index: int) -> str:
        return (
            f"Genera i dati di configurazione per un caso giudiziario italiano (numero caso: {index}). "
            "Scegli tra casi penali (frode, appropriazione indebita, corruzione, evasione fiscale), "
            "civili (inadempimento contrattuale, risarcimento danni) o amministrativi. "
            "Includi nome del caso, tribunale, data, parti in causa (imputato, pubblico ministero, giudice), "
            "capi di accusa, 3-5 fatti chiave, verdetto e condanna — tutto in italiano."
        )

    def article_user(self, config: BaseModel) -> str:
        config_json = config.model_dump_json(indent=2)
        return (
            "Sulla base della seguente configurazione del caso, crea una sentenza giudiziaria dettagliata di 800-1200 parole in italiano. "
            "Includi sintesi del caso, descrizione dettagliata dei fatti, esame delle prove, "
            "ragionamento giuridico, dispositivo e motivazione della condanna. "
            "Scrivi interamente in italiano.\n\n"
            f"Configurazione:\n{config_json}"
        )

    def factual_query_user(self, article: str) -> str:
        return (
            "Dalla sentenza italiana seguente, genera una domanda fattuale. "
            "La domanda deve avere risposta in una sola frase della sentenza. "
            "Includi la domanda, la risposta attesa e 1-2 citazioni esatte dalla sentenza. "
            "Scrivi tutto in italiano.\n\n"
            f"Sentenza:\n{article}"
        )

    def multihop_query_user(self, article: str) -> str:
        return (
            "Dalla sentenza italiana seguente, genera una domanda di ragionamento multi-salto. "
            "La domanda deve richiedere di collegare 2 o più informazioni distinte. "
            "Includi la domanda, una risposta dettagliata e 2 o più citazioni esatte. "
            "Scrivi tutto in italiano.\n\n"
            f"Sentenza:\n{article}"
        )

    def summarization_query_user(self, article: str) -> str:
        return (
            "Leggi l'intera sentenza italiana seguente e genera una domanda di sintesi. "
            "La domanda deve richiedere una comprensione dell'intero documento. "
            "Includi la domanda, una risposta esaustiva e 2-3 citazioni chiave. "
            "Scrivi tutto in italiano.\n\n"
            f"Sentenza:\n{article}"
        )

    def unanswerable_query_user(self, existing_entities: list[str]) -> str:
        entities_str = ", ".join(existing_entities[:15]) if existing_entities else "nessuno"
        return (
            "Genera una domanda plausibile relativa al diritto italiano. "
            "La domanda deve riguardare un caso o una persona fittizia NON presente nell'elenco seguente. "
            "La domanda deve sembrare autentica. "
            "Scrivi solo la domanda in italiano (non serve la risposta).\n\n"
            f"Casi/persone esistenti (usa qualcosa di diverso): {entities_str}"
        )

    def refine_user(self, question: str, answer: str, article: str) -> str:
        return (
            "Estrai dalla sentenza citazioni esatte a supporto della domanda e della risposta seguenti. "
            "Le citazioni devono corrispondere esattamente al testo della sentenza. "
            "Estrai 1-3 citazioni più pertinenti.\n\n"
            f"Domanda: {question}\nRisposta: {answer}\n\nSentenza:\n{article}"
        )
