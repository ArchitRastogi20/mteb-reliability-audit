"""
N-gram contamination check for the unified Multilingual RAG benchmark.

Question: are the GPT-5.4-generated RAG corpus passages substantively present in
publicly available text that the evaluated embedding models likely saw during
pretraining? We use:

(a) Wikipedia (per language) — proxy for the most universally consumed
    pretraining source. If our 8-grams overlap heavily with Wikipedia, the
    contamination concern is real; if rare, the corpus is distinctive.

(b) BelebeleRetrieval corpus (per language) — the same task we already use
    in §4 of the paper. Overlap here would imply our generated text is
    paraphrasing Belebele passages, which would invalidate the controlled
    probe.

For each of 6 RAG configs:
- Build the set of distinct 8-grams (whitespace-tokenised) in the corpus.
- Build the same for a 1k-doc sample of Wikipedia in the corresponding
  language.
- Build the same for the BelebeleRetrieval corpus for the corresponding
  language.
- Report:
  * |corpus 8-grams|
  * |corpus 8-grams ∩ Wikipedia 8-grams| / |corpus|  (% overlap)
  * |corpus 8-grams ∩ Belebele 8-grams| / |corpus|

Also record per-config self-stats: # docs, mean tokens/doc, # distinct 8-grams.

Output: analysis_output/stats/contamination_ngram.txt + a small CSV.

Runtime: ~2-5 minutes (Wikipedia download + tokenisation).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]  # anon_repo root
RAG_BASE = ROOT / "rag-dataset" / "output"
STATS = ROOT / "analysis_output" / "stats"
N = 8  # 8-gram

LANG_WIKI = {
    "it": "20231101.it",
    "ja": "20231101.ja",
    "hi": "20231101.hi",
}
LANG_BELEBELE = {
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
}
RAG_CONFIGS = [
    ("it", "finance"), ("it", "law"),
    ("ja", "finance"), ("ja", "law"),
    ("hi", "finance"), ("hi", "law"),
]
WIKI_SAMPLE_SIZE = 1000


def tokenize(text: str) -> list[str]:
    """Whitespace + punctuation-collapse tokeniser. Works for Latin and Devanagari.
    For Japanese (no spaces), we fall back to character n-grams below."""
    # Keep letters, digits; replace everything else with whitespace.
    cleaned = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return cleaned.split()


def char_ngrams(text: str, n: int = N) -> Iterable[str]:
    """Character n-grams (use for Japanese where word-tokenisation is fraught)."""
    text = re.sub(r"\s+", " ", text)
    for i in range(len(text) - n + 1):
        yield text[i : i + n]


def ngrams(text: str, n: int = N, lang: str = "it") -> set[str]:
    if lang == "ja":
        # Japanese has no whitespace; use character 8-grams
        return set(char_ngrams(text, n))
    toks = tokenize(text)
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def load_rag_corpus(lang: str, domain: str) -> list[str]:
    p = RAG_BASE / lang / domain / "corpus.jsonl"
    docs = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(row.get("text", ""))
    return docs


def load_wiki_sample(lang: str, n: int = WIKI_SAMPLE_SIZE) -> list[str]:
    """Sample n Wikipedia docs for the given language. Uses the wikimedia/wikipedia
    HF dataset (much smaller than full Wikipedia download — ~50MB streaming)."""
    from datasets import load_dataset
    print(f"  loading wikipedia/{LANG_WIKI[lang]} (streaming)...")
    ds = load_dataset(
        "wikimedia/wikipedia", LANG_WIKI[lang], split="train", streaming=True
    )
    docs: list[str] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        docs.append(row.get("text", ""))
    return docs


def load_belebele_corpus(lang_code: str) -> list[str]:
    from datasets import load_dataset
    print(f"  loading mteb/belebele/{lang_code}...")
    ds = load_dataset("mteb/belebele", lang_code, split="test")
    seen: dict[str, str] = {}
    for row in ds:
        link = row.get("link", "")
        if link and link not in seen:
            seen[link] = row.get("flores_passage", "")
    return list(seen.values())


def overlap_pct(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return 100.0 * len(a & b) / len(a)


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)

    # Pre-load Wikipedia + Belebele per language (3 langs total)
    wiki_ngrams: dict[str, set[str]] = {}
    bele_ngrams: dict[str, set[str]] = {}
    for lang in ["it", "ja", "hi"]:
        print(f"[{lang}] loading reference corpora")
        wiki_docs = load_wiki_sample(lang)
        wiki_ng = set()
        for d in wiki_docs:
            wiki_ng |= ngrams(d, N, lang=lang)
        wiki_ngrams[lang] = wiki_ng

        bele_docs = load_belebele_corpus(LANG_BELEBELE[lang])
        bele_ng = set()
        for d in bele_docs:
            bele_ng |= ngrams(d, N, lang=lang)
        bele_ngrams[lang] = bele_ng
        print(f"  wiki {lang}: {len(wiki_ng):,} unique 8-grams "
              f"from {len(wiki_docs)} docs")
        print(f"  bele {lang}: {len(bele_ng):,} unique 8-grams "
              f"from {len(bele_docs)} docs")

    lines = ["# N-gram contamination check — RAG corpus vs Wikipedia + Belebele\n",
             "Method: 8-gram set overlap. Word-level for IT/HI (whitespace token),",
             "character-level for JA (no word boundaries).",
             "Reference corpora:",
             f"  - Wikipedia: streaming first {WIKI_SAMPLE_SIZE} docs per language",
             f"    (wikimedia/wikipedia 20231101 subset)",
             f"  - BelebeleRetrieval: full corpus (~488 unique passages per language)",
             "",
             f"{'Config':<12s}{'#docs':>7s}{'#8gr':>10s}{'wiki%':>8s}{'bele%':>8s}",
             "-" * 50]

    rows = []
    for lang, domain in RAG_CONFIGS:
        ds = f"{lang}_{domain}"
        docs = load_rag_corpus(lang, domain)
        rag_ng: set[str] = set()
        for d in docs:
            rag_ng |= ngrams(d, N, lang=lang)
        wiki_pct = overlap_pct(rag_ng, wiki_ngrams[lang])
        bele_pct = overlap_pct(rag_ng, bele_ngrams[lang])
        lines.append(
            f"{ds:<12s}{len(docs):>7d}{len(rag_ng):>10,d}"
            f"{wiki_pct:>7.2f}%{bele_pct:>7.2f}%"
        )
        rows.append(dict(
            config=ds, n_docs=len(docs), n_ngrams=len(rag_ng),
            wiki_pct=round(wiki_pct, 2), bele_pct=round(bele_pct, 2),
        ))

    avg_wiki = sum(r["wiki_pct"] for r in rows) / len(rows)
    avg_bele = sum(r["bele_pct"] for r in rows) / len(rows)
    lines += ["",
              f"AVERAGE Wikipedia 8-gram overlap: {avg_wiki:.2f}%",
              f"AVERAGE Belebele  8-gram overlap: {avg_bele:.2f}%",
              "",
              "INTERPRETATION:",
              "  - Wikipedia overlap < 5% indicates the GPT-5.4-generated corpus",
              "    is substantively distinct from web-Wikipedia text the evaluated",
              "    embedding models likely saw during pretraining.",
              "  - Belebele overlap < 1% confirms our RAG corpus is not paraphrasing",
              "    the Belebele probe (which would invalidate §4.5's controlled probe).",
              "  - Higher overlap on common-language n-grams (function words, dates,",
              "    common entities) is expected and not a contamination signal.",
              "",
              "CAVEATS:",
              "  - 8-gram exact-match is conservative; semantic paraphrase can pass",
              "    this filter and still be considered contamination by some reviewers.",
              "  - Wikipedia is a proxy for the larger web pretraining corpora",
              "    (mC4, CommonCrawl) we cannot fully replicate here.",
              "  - For Japanese we use character 8-grams (no word boundaries), giving",
              "    higher overlap baselines; compare JA configs to JA Wikipedia row only."]

    out_path = STATS / "contamination_ngram.txt"
    out_path.write_text("\n".join(lines))
    print(f"\n[done] wrote {out_path}")
    print("\n".join(lines[-15:]))

    # CSV
    import csv
    csv_path = STATS / "contamination_ngram.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n_docs", "n_ngrams",
                                           "wiki_pct", "bele_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
