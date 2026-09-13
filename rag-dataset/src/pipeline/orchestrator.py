from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal

# Make `src` importable when running as a script from any directory
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env from project root (CHATGPT_API_KEY or OPENAI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from src.client.async_openai import AsyncOpenAIClient, CostTracker
from src.config import PipelineConfig
from src.evaluation.bm25_sanity import run_bm25_sanity
from src.pipeline.beir_writer import write_beir_dataset
from src.pipeline.stage1_config import run_stage1
from src.pipeline.stage2_article import run_stage2
from src.pipeline.stage3_queries import run_stage3
from src.pipeline.stage4_refine import run_stage4
from src.pipeline.stage5_answer import run_stage5
from src.validation.answer_validator import run_answer_validation
from src.validation.llm_validator import run_validation

LANGUAGES: list[Literal["ja", "hi", "it"]] = ["ja", "hi", "it"]
DOMAINS: list[Literal["finance", "law"]] = ["finance", "law"]


async def run_pipeline(
    lang: Literal["ja", "hi", "it"],
    domain: Literal["finance", "law"],
    count: int,
    client: AsyncOpenAIClient,
    checkpoint_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    stage: int | None = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"Pipeline: lang={lang}, domain={domain}, count={count}")
    print(f"{'='*60}")

    if stage is None or stage == 1:
        print("\n[Stage 1] Generating configs...")
        configs = await run_stage1(lang, domain, count, client, checkpoint_dir, semaphore_limit=cfg.semaphores.stage1)
        print(f"  -> {len(configs)} configs generated")
    else:
        # Load from checkpoint for later stages
        import json
        from src.models.config_models import FinanceConfig, LawConfig
        ckpt = checkpoint_dir / f"configs_{lang}_{domain}.jsonl"
        model = FinanceConfig if domain == "finance" else LawConfig
        configs = []
        if ckpt.exists():
            with ckpt.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        item.pop("_index", None)
                        configs.append(model(**item))
        print(f"  -> Loaded {len(configs)} configs from checkpoint")

    if stage is not None and stage < 2:
        print(f"\n[Cost so far] {client.cost_tracker.summary()}")
        return

    if stage is None or stage == 2:
        print("\n[Stage 2] Generating articles...")
        documents = await run_stage2(lang, domain, configs, client, checkpoint_dir, semaphore_limit=cfg.semaphores.stage2)
        print(f"  -> {len(documents)} articles generated")
    else:
        import json
        from src.models.corpus_models import Document
        ckpt = checkpoint_dir / f"articles_{lang}_{domain}.jsonl"
        documents = []
        if ckpt.exists():
            with ckpt.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        item.pop("_index", None)
                        documents.append(Document(**item))
        print(f"  -> Loaded {len(documents)} documents from checkpoint")

    if stage is not None and stage < 3:
        print(f"\n[Cost so far] {client.cost_tracker.summary()}")
        return

    if stage is None or stage == 3:
        print("\n[Stage 3] Generating queries...")
        queries = await run_stage3(lang, domain, documents, configs, client, checkpoint_dir, semaphore_limit=cfg.semaphores.stage3)
        print(f"  -> {len(queries)} queries generated")
    else:
        import json
        from src.models.query_models import QARPair
        ckpt = checkpoint_dir / f"queries_{lang}_{domain}.jsonl"
        queries = []
        if ckpt.exists():
            with ckpt.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        queries.append(QARPair(**json.loads(line)))
        print(f"  -> Loaded {len(queries)} queries from checkpoint")

    if stage is not None and stage < 4:
        print(f"\n[Cost so far] {client.cost_tracker.summary()}")
        return

    if stage is None or stage == 4:
        print("\n[Stage 4] Refining references...")
        queries = await run_stage4(lang, domain, queries, documents, client, checkpoint_dir, semaphore_limit=cfg.semaphores.stage4)
        print(f"  -> {len(queries)} queries refined")

    print("\n[Validation] Running LLM ground-truth validation...")
    queries = await run_validation(queries, documents, client, semaphore_limit=cfg.semaphores.validation)
    valid_count = sum(1 for q in queries if q.valid)
    print(f"  -> {valid_count}/{len(queries)} queries valid")

    print("\n[Stage 5] Generating RAG answers + extracting keypoints...")
    queries = await run_stage5(lang, domain, queries, client, checkpoint_dir, semaphore_limit=cfg.semaphores.stage5)
    answered = sum(1 for q in queries if q.generated_answer)
    print(f"  -> {answered} answers generated")

    print("\n[Answer Validation] Checking answer coverage against keypoints...")
    queries = await run_answer_validation(queries, client)
    ans_valid = sum(1 for q in queries if q.valid and q.answer_valid and q.query_type != "unanswerable")
    answerable = sum(1 for q in queries if q.query_type != "unanswerable" and q.valid)
    print(f"  -> {ans_valid}/{answerable} generated answers pass coverage check")

    print("\n[Output] Writing BEIR dataset...")
    write_beir_dataset(lang, domain, documents, queries, output_dir)
    print(f"  -> corpus.jsonl, queries.jsonl, full_queries.jsonl, qrels/test.tsv")
    print(f"  -> Written to {output_dir / lang / domain}/")

    print("\n[BM25] Running sanity check...")
    result = run_bm25_sanity(lang, domain, output_dir, output_dir / "evaluation")
    ndcg10 = result.get("metrics", {}).get("ndcg_cut_10", 0)
    interp = result.get("ndcg10_interpretation", "unknown")
    print(f"  -> NDCG@10={ndcg10:.4f} ({interp})")

    print(f"\n[Done] {client.cost_tracker.summary()}")


def _dry_run(cfg: PipelineConfig) -> None:
    print("=== DRY RUN — Cost Estimate ===")
    print(f"Docs per combo: {cfg.corpus.docs_per_combo} × 4 combos = {cfg.corpus.docs_per_combo * 4} total")
    print(f"Queries per combo: {cfg.queries.total} × 4 combos = {cfg.queries.total * 4} total")
    print(f"  factual={cfg.queries.factual}, multi_hop={cfg.queries.multi_hop}, "
          f"summarization={cfg.queries.summarization}, unanswerable={cfg.queries.unanswerable}")
    print(f"Budget cap: ${cfg.budget_usd:.2f}")
    print()
    print("Stage 1 (config, nano):         ~$0.50")
    print("Stage 2 (article, mini):        ~$5.00")
    print("Stage 3 factual (nano):         ~$1.00")
    print("Stage 3 multi-hop (mini):       ~$4.20")
    print("Stage 3 summarization (mini):   ~$2.40")
    print("Stage 3 unanswerable (nano):    ~$0.20")
    print("Stage 4 refinement (nano):      ~$1.40")
    print("Validation (nano):              ~$1.00")
    print("Stage 5 RAG answers (mini):     ~$3.50")
    print("Stage 5 keypoints (nano):       ~$0.50")
    print("Answer validation (nano):       ~$0.80")
    print("-----------------------------------")
    print("Total estimate (base):          ~$20.00")
    print("Total estimate (+20% buffer):   ~$24.00")
    print(f"Hard budget cap:                ${cfg.budget_usd:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Dataset Generation Pipeline")
    parser.add_argument("--lang", choices=["ja", "hi", "it"], help="Target language")
    parser.add_argument("--domain", choices=["finance", "law"], help="Target domain")
    parser.add_argument("--all", action="store_true", help="Run all 4 lang×domain combinations")
    parser.add_argument("--dry-run", action="store_true", help="Print cost estimate, no API calls")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Run single stage only")
    parser.add_argument("--count", type=int, default=None, help="Docs per domain per language (default: from config.yaml)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = PipelineConfig.load()

    if args.dry_run:
        _dry_run(cfg)
        return

    if args.all:
        combos = [(lang, domain) for lang in LANGUAGES for domain in DOMAINS]
    elif args.lang and args.domain:
        combos = [(args.lang, args.domain)]
    else:
        parser.error("Provide --lang and --domain, or use --all")
        return

    tracker = CostTracker(budget=cfg.budget_usd)
    client = AsyncOpenAIClient(tracker)

    async def run_all():
        count = args.count or cfg.corpus.docs_per_combo
        checkpoint_dir = args.checkpoint_dir or Path(cfg.paths.checkpoint_dir)
        output_dir = args.output_dir or Path(cfg.paths.output_dir)
        for lang, domain in combos:
            await run_pipeline(
                lang=lang,
                domain=domain,
                count=count,
                client=client,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                cfg=cfg,
                stage=args.stage,
            )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
