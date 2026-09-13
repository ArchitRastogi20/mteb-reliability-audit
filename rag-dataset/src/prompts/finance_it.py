from __future__ import annotations

from pydantic import BaseModel

from src.prompts import PromptSet


class FinanceItPrompts(PromptSet):
    language = "it"
    domain = "finance"

    @property
    def config_system(self) -> str:
        return "Sei un esperto nella generazione di dati su aziende finanziarie italiane. Genera dati di configurazione aziendale nel formato specificato."

    @property
    def article_system(self) -> str:
        return "Sei un esperto nella redazione di relazioni finanziarie in italiano. Sulla base della configurazione aziendale fornita, crea una relazione annuale dettagliata interamente in italiano."

    @property
    def query_system(self) -> str:
        return "Sei un esperto nella generazione di domande per la valutazione RAG. Genera domande nel formato esatto richiesto seguendo le istruzioni."

    @property
    def refine_system(self) -> str:
        return "Sei un esperto nell'estrazione di citazioni esatte da documenti. Le citazioni devono corrispondere esattamente al testo del documento."

    def config_user(self, index: int) -> str:
        return (
            f"Genera i dati di configurazione per un'azienda finanziaria italiana (numero azienda: {index}). "
            "Scegli tra settori come banca, assicurazione, leasing, gestione patrimoniale, borsa valori, credito al consumo. "
            "Includi nome dell'azienda, settore, anno di fondazione, sede, 2-3 eventi importanti (data, descrizione, impatto finanziario), "
            "fatturato e margine di profitto — tutto in italiano."
        )

    def article_user(self, config: BaseModel) -> str:
        config_json = config.model_dump_json(indent=2)
        return (
            "Sulla base della seguente configurazione aziendale, crea una relazione annuale dettagliata di 800-1200 parole in italiano. "
            "Includi panoramica aziendale, risultati finanziari, descrizione dettagliata degli eventi chiave, analisi di mercato e prospettive future. "
            "Scrivi interamente in italiano, senza usare l'inglese.\n\n"
            f"Configurazione:\n{config_json}"
        )

    def factual_query_user(self, article: str) -> str:
        return (
            "Dall'articolo italiano seguente, genera una domanda fattuale. "
            "La domanda deve avere risposta in una sola frase dell'articolo. "
            "Includi la domanda, la risposta attesa e 1-2 citazioni esatte dall'articolo a supporto. "
            "Scrivi tutto in italiano.\n\n"
            f"Articolo:\n{article}"
        )

    def multihop_query_user(self, article: str) -> str:
        return (
            "Dall'articolo italiano seguente, genera una domanda di ragionamento multi-salto. "
            "La domanda deve richiedere di collegare 2 o più informazioni distinte presenti nell'articolo. "
            "Includi la domanda, una risposta dettagliata e 2 o più citazioni esatte. "
            "Scrivi tutto in italiano.\n\n"
            f"Articolo:\n{article}"
        )

    def summarization_query_user(self, article: str) -> str:
        return (
            "Leggi l'intero articolo italiano seguente e genera una domanda di sintesi. "
            "La domanda deve richiedere una comprensione dell'intero documento. "
            "Includi la domanda, una risposta esaustiva e 2-3 citazioni chiave. "
            "Scrivi tutto in italiano.\n\n"
            f"Articolo:\n{article}"
        )

    def unanswerable_query_user(self, existing_entities: list[str]) -> str:
        entities_str = ", ".join(existing_entities[:15]) if existing_entities else "nessuna"
        return (
            "Genera una domanda plausibile relativa al settore finanziario italiano. "
            "La domanda deve riguardare un'azienda fittizia NON presente nell'elenco seguente. "
            "La domanda deve sembrare autentica, come se riguardasse una vera azienda. "
            "Scrivi solo la domanda in italiano (non serve la risposta).\n\n"
            f"Aziende esistenti (usa un'azienda diversa): {entities_str}"
        )

    def refine_user(self, question: str, answer: str, article: str) -> str:
        return (
            "Estrai dall'articolo citazioni esatte a supporto della domanda e della risposta seguenti. "
            "Le citazioni devono corrispondere esattamente al testo dell'articolo. "
            "Estrai 1-3 citazioni più pertinenti.\n\n"
            f"Domanda: {question}\nRisposta: {answer}\n\nArticolo:\n{article}"
        )
