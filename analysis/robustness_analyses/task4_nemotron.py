#!/usr/bin/env python3
"""Task 4: Nemotron-8b Italian query-type triangulation."""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260506
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent      # repo root
OUT_DIR = SCRIPT_DIR / "task4_nemotron_triangulation"
OUT_DIR.mkdir(exist_ok=True)

RAG_DIR = REPO_ROOT / "rag-dataset" / "output" / "evaluation"

logging.basicConfig(
    filename=str(OUT_DIR / "task4_nemotron_triangulation.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# Verified by reading model_id field from each JSON
TARGET_MODELS = {
    "nvidia/llama-embed-nemotron-8b": "llama_embed_nemotron_eeabda",
    "intfloat/multilingual-e5-small": "multilingual_e5_smal_6f25f9",
    "intfloat/multilingual-e5-large": "multilingual_e5_larg_e3a0cc",
    "Qwen/Qwen3-Embedding-4B": "Qwen3_Embedding_4B_8f1f6c",
}

MODEL_SHORT = {
    "nvidia/llama-embed-nemotron-8b": "nemotron-8b",
    "intfloat/multilingual-e5-small": "ml-e5-small",
    "intfloat/multilingual-e5-large": "ml-e5-large",
    "Qwen/Qwen3-Embedding-4B": "Qwen3-4B",
}

CONFIGS = ["it_finance", "it_law"]
QUERY_TYPES = ["factual", "multi_hop", "summarization"]


def load_by_type(model_id: str, config: str) -> dict:
    model_dir = TARGET_MODELS[model_id]
    f = RAG_DIR / model_dir / f"{config}.json"
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)["by_type"]


def main():
    t0 = time.time()
    log.info(f"RAG eval dir: {RAG_DIR}")
    log.info(f"Target models: {list(MODEL_SHORT.values())}")
    log.info(f"Configs: {CONFIGS}")
    log.info(f"Query types: {QUERY_TYPES} (unanswerable excluded)")
    log.info(
        "Data source: by_type aggregates from evaluation JSONs. "
        "Per-query NDCG not stored -> bootstrap CIs over queries not computable. "
        "Reporting point estimates only."
    )

    rows = []
    for model_id, model_short in MODEL_SHORT.items():
        for config in CONFIGS:
            try:
                by_type = load_by_type(model_id, config)
                log.info(f"{model_short}/{config}: loaded, types={list(by_type.keys())}")
            except FileNotFoundError:
                log.warning(f"Missing: {TARGET_MODELS[model_id]}/{config}.json")
                continue

            for qtype in QUERY_TYPES:
                if qtype not in by_type:
                    log.warning(f"Missing qtype '{qtype}' in {model_short}/{config}")
                    continue
                ndcg = by_type[qtype]["ndcg_at_10"] * 100
                log.info(f"  {model_short}/{config}/{qtype}: NDCG@10={ndcg:.2f}")
                rows.append(
                    dict(
                        config=config,
                        model=model_short,
                        query_type=qtype,
                        n_queries="N/A",
                        mean_ndcg10=round(ndcg, 2),
                        ci_lo="N/A",
                        ci_hi="N/A",
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "nemotron_querytype.csv", index=False)
    log.info(f"Saved nemotron_querytype.csv ({len(df)} rows)")

    # Gap table: nemotron vs each control per (config, query_type)
    gap_rows = []
    for config in CONFIGS:
        for qtype in QUERY_TYPES:
            def val(model):
                r = df[(df.model == model) & (df.config == config) & (df.query_type == qtype)]
                return float(r.iloc[0].mean_ndcg10) if not r.empty else None

            nem = val("nemotron-8b")
            small = val("ml-e5-small")
            large = val("ml-e5-large")
            qwen = val("Qwen3-4B")

            if nem is None or small is None:
                log.warning(f"Missing data for gap table: {config}/{qtype}")
                continue

            gap_rows.append(
                dict(
                    config=config,
                    query_type=qtype,
                    nemotron_mean=nem,
                    ml_e5_small_mean=small,
                    ml_e5_large_mean=large,
                    qwen3_4b_mean=qwen,
                    gap_vs_small=round(nem - small, 2),
                    gap_ci_lo="N/A",
                    gap_ci_hi="N/A",
                )
            )

    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(OUT_DIR / "nemotron_querytype_gap.csv", index=False)
    log.info(f"Saved nemotron_querytype_gap.csv ({len(gap_df)} rows)")

    if gap_df.empty:
        md = (
            "## Nemotron-8b Italian Query-Type Triangulation (App. N / Section 4.4)\n\n"
            "**No data loaded.** Check RAG eval files exist for target models.\n"
        )
        (OUT_DIR / "nemotron_triangulation.md").write_text(md, encoding="utf-8")
        log.info(f"Done in {time.time() - t0:.1f}s")
        return

    # Compute average gap per query type across configs
    avg_gap = (
        gap_df.groupby("query_type")["gap_vs_small"]
        .mean()
        .reindex(QUERY_TYPES)
    )
    log.info(f"Mean gap (nemotron - ml-e5-small) by query type: {avg_gap.to_dict()}")

    factual_g = float(avg_gap.get("factual", float("nan")))
    mhop_g = float(avg_gap.get("multi_hop", float("nan")))
    sum_g = float(avg_gap.get("summarization", float("nan")))

    all_gaps = [g for g in [factual_g, mhop_g, sum_g] if not np.isnan(g)]
    spread = max(all_gaps) - min(all_gaps) if len(all_gaps) >= 2 else 0.0

    # All gaps are negative (nemotron underperforms); min = biggest deficit.
    if spread < 5.0:
        pattern = "uniform"
        interp = (
            "The performance gap between nemotron-8b and ml-e5-small is roughly uniform "
            "across query types, indicating the Italian failure is not confined to one "
            "retrieval modality. The domain-coverage hypothesis is not further refined "
            "by query type."
        )
    elif factual_g == min(all_gaps):
        pattern = "concentrated on factual queries"
        interp = (
            "The gap is largest on factual queries, consistent with the hypothesized "
            "domain-coverage shortfall on encyclopedic Italian text (Mechanism B). "
            "This supports but does not confirm the hypothesis without public training-data statistics."
        )
    elif mhop_g == min(all_gaps):
        pattern = "concentrated on multi-hop queries"
        interp = (
            "The gap is largest on multi-hop queries. This is not clearly predicted by "
            "the encyclopedic domain-coverage hypothesis and warrants further investigation."
        )
    else:
        pattern = "concentrated on summarization queries"
        interp = (
            "The gap is largest on summarization queries. This is not clearly predicted by "
            "the encyclopedic domain-coverage hypothesis."
        )

    # Markdown summary table
    table_lines = [
        "| config | query_type | nemotron-8b | ml-e5-small | ml-e5-large | Qwen3-4B | gap (nem - small) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in gap_df.iterrows():
        def fmt(v):
            return f"{v:.1f}" if v is not None else "N/A"
        table_lines.append(
            f"| {r.config} | {r.query_type} | {fmt(r.nemotron_mean)} | "
            f"{fmt(r.ml_e5_small_mean)} | {fmt(r.ml_e5_large_mean)} | "
            f"{fmt(r.qwen3_4b_mean)} | {r.gap_vs_small:+.1f} |"
        )

    md = (
        "## Nemotron-8b Italian Query-Type Triangulation (App. N / Section 4.4)\n\n"
        "Beyond the max_seq_length sweep (App. N), we break nemotron-8b's IT-RAG "
        "performance down by query type on it-fin/it-law, comparing to a small control "
        "(ml-e5-small) and two strong controls (ml-e5-large, Qwen3-4B). "
        f"The gap to ml-e5-small is {pattern} "
        f"(factual: {factual_g:+.1f} pp, "
        f"multi-hop: {mhop_g:+.1f} pp, "
        f"summarization: {sum_g:+.1f} pp, averaged over it-fin/it-law). "
        f"{interp} "
        "Bootstrap CIs over individual queries are not reported as evaluation outputs "
        "store by-type aggregates rather than per-query NDCG scores.\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        "*NDCG@10 x100. Gap: nemotron-8b minus ml-e5-small (positive = nemotron better).*\n"
    )
    (OUT_DIR / "nemotron_triangulation.md").write_text(md, encoding="utf-8")
    log.info(f"Saved nemotron_triangulation.md. Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
