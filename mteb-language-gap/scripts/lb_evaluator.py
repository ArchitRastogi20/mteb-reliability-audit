from __future__ import annotations

import gc
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ranx import Qrels, Run, evaluate
from sentence_transformers import SentenceTransformer

import mteb

# ── Runtime compatibility patches ────────────────────────────────────────────
# Patch 1: bge-m3 — torch 2.4 blocks torch.load (CVE-2025-32434).
# modeling_utils imports check_torch_load_is_safe by name, so we must patch
# the reference inside that module, not just in import_utils.
import transformers.modeling_utils as _tm
if hasattr(_tm, "check_torch_load_is_safe"):
    _tm.check_torch_load_is_safe = lambda: None

# Patch 2: pplx-embed — create_causal_mask(or_mask_function=...) requires
# torch>=2.6. When or_mask_function is present it signals bidirectional intent.
# create_bidirectional_mask returns None for non-padded sequences; Qwen3's SDPA
# then recomputes a causal mask internally. Return an explicit 4D all-zero float
# mask (0.0 = attend to all) so Qwen3 uses it directly without recomputing.
# Use explicit None checks — `or` on Tensors raises RuntimeError.
try:
    import transformers.masking_utils as _mu
    _orig_ccm = _mu.create_causal_mask
    def _patched_ccm(*args, **kwargs):
        or_fn = kwargs.pop("or_mask_function", None)
        kwargs.pop("and_mask_function", None)
        if or_fn is not None:
            embs = kwargs.get("inputs_embeds")
            if embs is None:
                embs = kwargs.get("input_embeds")
            if embs is None and len(args) > 1:
                embs = args[1]
            if embs is not None:
                batch_size, seq_len = embs.shape[0], embs.shape[1]
                return torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=embs.dtype,
                    device=embs.device,
                )
            return None
        return _orig_ccm(*args, **kwargs)
    _mu.create_causal_mask = _patched_ccm
except Exception:
    pass

# Patch 3: gte-Qwen2-7B — custom modeling_qwen.py accesses config.rope_theta
# which is absent in this transformers version's Qwen2Config.
try:
    from transformers import Qwen2Config as _Q2C
    _q2c_orig_ga = _Q2C.__getattribute__
    def _q2c_getattribute(self, name):
        if name == "rope_theta":
            try:
                return _q2c_orig_ga(self, name)
            except AttributeError:
                return 10000.0
        return _q2c_orig_ga(self, name)
    _Q2C.__getattribute__ = _q2c_getattribute
except Exception:
    pass

# Patch 4: gte-Qwen2-7B — DynamicCache methods removed in transformers 4.47+.
try:
    from transformers import DynamicCache as _DC
    if not hasattr(_DC, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is None:
                return cache
            for layer_past in past_key_values:
                cache.update(layer_past[0], layer_past[1], len(cache.key_cache))
            return cache
        _DC.from_legacy_cache = _from_legacy_cache
    if not hasattr(_DC, "get_usable_length"):
        def _dc_get_usable_length(self, new_seq_length: int = 0, layer_idx: int = 0) -> int:
            max_length = self.get_max_length() if hasattr(self, "get_max_length") else None
            seq_len = self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
            if max_length is not None:
                return min(max_length - new_seq_length, seq_len)
            return seq_len
        _DC.get_usable_length = _dc_get_usable_length
    if not hasattr(_DC, "to_legacy_cache"):
        def _dc_to_legacy_cache(self):
            if not self.key_cache:
                return None
            return tuple(
                (self.key_cache[i], self.value_cache[i])
                for i in range(len(self.key_cache))
            )
        _DC.to_legacy_cache = _dc_to_legacy_cache
except Exception:
    pass

# Patch 5: jina-v3 — XLMRobertaLoRA (custom class) is missing all_tied_weights_keys,
# a dict attribute that transformers 5.x _move_missing_keys_from_meta_to_device expects.
try:
    import transformers.modeling_utils as _tmmu
    _orig_mmk = _tmmu.PreTrainedModel._move_missing_keys_from_meta_to_device
    def _patched_mmk(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_mmk(self, *args, **kwargs)
    _tmmu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_mmk
except Exception:
    pass

from scripts.embedding_cache import EmbeddingCache
from scripts.utils import log_json, sanitize_model_id, vram_peak_mb

REPO_ROOT = Path(__file__).parent.parent
CHROMA_PATH = "/workspace/chroma_cache"
DATASET_CACHE_DIR = Path("/workspace/dataset_cache")

logger = logging.getLogger("mteb_lb")


# ── Language Configurations ───────────────────────────────────────────────────

@dataclass
class LanguageConfig:
    lang_code: str
    tasks: list[str]
    task_config: dict[str, dict]
    task_short: dict[str, str]
    results_base: Path
    avg_label: str
    mteb_csv: Path


LANG_CONFIGS: dict[str, LanguageConfig] = {
    "jpn": LanguageConfig(
        lang_code="jpn",
        tasks=["BelebeleRetrieval", "MIRACLRetrievalHardNegatives"],
        task_config={
            "BelebeleRetrieval":            {"hf_subset": "jpn_Jpan-jpn_Jpan", "split": "test"},
            "MIRACLRetrievalHardNegatives": {"hf_subset": "ja",                "split": "dev"},
        },
        task_short={
            "BelebeleRetrieval":            "jblb",
            "MIRACLRetrievalHardNegatives": "jmhn",
        },
        results_base=REPO_ROOT / "results" / "jpn_lb",
        avg_label="AvgJPN",
        mteb_csv=REPO_ROOT.parent / "japanese_metb" / "japanese_performance_per_task.csv",
    ),
    "hin": LanguageConfig(
        lang_code="hin",
        tasks=[
            "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            "MLQARetrieval",
            "WikipediaRetrievalMultilingual",
        ],
        task_config={
            "BelebeleRetrieval":            {"hf_subset": "hin_Deva-hin_Deva", "split": "test"},
            "MIRACLRetrievalHardNegatives": {"hf_subset": "hi",                "split": "dev"},
            "MLQARetrieval":                {"hf_subset": "hin-hin",           "split": "test"},
            "WikipediaRetrievalMultilingual": {"hf_subset": "hi",              "split": "test"},
        },
        task_short={
            "BelebeleRetrieval":            "hblb",
            "MIRACLRetrievalHardNegatives": "hmhn",
            "MLQARetrieval":                "hmlq",
            "WikipediaRetrievalMultilingual": "hwki",
        },
        results_base=REPO_ROOT / "results" / "hin_lb",
        avg_label="AvgHIN",
        mteb_csv=REPO_ROOT.parent / "analysis" / "mteb_csvs" / "hindi_mteb" / "hindi_performance_per_task.csv",
    ),
    "ita": LanguageConfig(
        lang_code="ita",
        tasks=[
            "BelebeleRetrieval",
            "WikipediaRetrievalMultilingual",
        ],
        task_config={
            "BelebeleRetrieval":              {"hf_subset": "ita_Latn-ita_Latn", "split": "test"},
            "WikipediaRetrievalMultilingual": {"hf_subset": "it",                "split": "test"},
        },
        task_short={
            "BelebeleRetrieval":              "iblb",
            "WikipediaRetrievalMultilingual": "iwki",
        },
        results_base=REPO_ROOT / "results" / "ita_lb",
        avg_label="AvgITA",
        mteb_csv=REPO_ROOT.parent / "analysis" / "mteb_csvs" / "italian_mteb" / "italian_performance_per_task.csv",
    ),
    "swa": LanguageConfig(
        lang_code="swa",
        tasks=["BelebeleRetrieval"],
        task_config={
            "BelebeleRetrieval": {"hf_subset": "swh_Latn-swh_Latn", "split": "test"},
        },
        task_short={
            "BelebeleRetrieval": "sblb",
        },
        results_base=REPO_ROOT / "results" / "swa_lb",
        avg_label="AvgSWA",
        mteb_csv=REPO_ROOT.parent / "swahili_mteb" / "swahili_mteb_performance_per_task.csv",
    ),
}


# ── Collection name helpers ───────────────────────────────────────────────────

def _corpus_col(model_id: str, task_name: str, cfg: LanguageConfig) -> str:
    short = cfg.task_short[task_name]
    return f"c_{sanitize_model_id(model_id)}_{short}"


def _query_col(model_id: str, task_name: str, cfg: LanguageConfig) -> str:
    short = cfg.task_short[task_name]
    return f"q_{sanitize_model_id(model_id)}_{short}"


# ── Task data loading ─────────────────────────────────────────────────────────

def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def _load_belebele(hf_subset: str) -> tuple[dict, dict, dict]:
    """Load BelebeleRetrieval for a monolingual subset (e.g. 'jpn_Jpan-jpn_Jpan').

    BelebeleRetrieval.load_data() is broken — it calls load_dataset without a
    config name. We load the single language config directly instead.
    """
    from datasets import load_dataset
    lang_config = hf_subset.split("-")[0]  # "jpn_Jpan-jpn_Jpan" → "jpn_Jpan"
    revision = "979a211276faa22f671e69d096634193567cfd05"
    ds = load_dataset("mteb/belebele", lang_config, revision=revision, split="test")

    link_to_cid: dict[str, str] = {}
    corpus: dict[str, str] = {}
    for row in ds:
        link = row["link"]
        if link not in link_to_cid:
            cid = f"C{len(link_to_cid)}"
            link_to_cid[link] = cid
            corpus[cid] = row["flores_passage"]

    question_to_qid: dict[str, str] = {}
    queries: dict[str, str] = {}
    for row in ds:
        q = row["question"]
        if q not in question_to_qid:
            qid = f"Q{len(question_to_qid)}"
            question_to_qid[q] = qid
            queries[qid] = q

    qrels: dict[str, dict[str, int]] = {}
    for row in ds:
        qid = question_to_qid[row["question"]]
        cid = link_to_cid[row["link"]]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][cid] = 1

    return corpus, queries, qrels


def _task_cache_path(task_name: str, hf_subset: str, split: str) -> Path:
    key = f"{task_name}__{hf_subset}__{split}".replace("/", "-")
    return DATASET_CACHE_DIR / f"{key}.json"


def _load_task_data(task_name: str, cfg: LanguageConfig) -> dict:
    task_cfg = cfg.task_config[task_name]
    hf_subset = task_cfg["hf_subset"]
    split = task_cfg["split"]

    cache_path = _task_cache_path(task_name, hf_subset, split)
    if cache_path.exists():
        logger.info("Dataset disk cache hit: %s", cache_path.name)
        cached = json.loads(cache_path.read_text())
        return {"task_name": task_name, **cached}

    if task_name == "BelebeleRetrieval":
        corpus, queries, qrels = _load_belebele(hf_subset)
    else:
        t = mteb.get_task(task_name)
        t.load_data(eval_splits=[split], hf_subsets=[hf_subset])
        data = t.dataset[hf_subset][split]
        corpus = {row["id"]: _corpus_text(row) for row in data["corpus"]}
        queries = {row["id"]: row["text"] for row in data["queries"]}
        qrels = {str(k): {str(dk): int(dv) for dk, dv in v.items()} for k, v in data["relevant_docs"].items()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"corpus": corpus, "queries": queries, "qrels": qrels}))
    logger.info("Dataset cached to disk: %s", cache_path.name)

    logger.debug(
        "Loaded %s [%s/%s]: corpus=%d, queries=%d",
        task_name, hf_subset, split, len(corpus), len(queries),
    )
    return {"task_name": task_name, "corpus": corpus, "queries": queries, "qrels": qrels}


# ── NDCG computation ──────────────────────────────────────────────────────────

def _compute_ndcg10(
    corpus_ids: list[str],
    corpus_embs: np.ndarray,
    query_ids: list[str],
    query_embs: np.ndarray,
    qrels: dict,
    task_name: str,
) -> tuple[str, float]:
    scores = query_embs @ corpus_embs.T
    top_k = min(10, scores.shape[1])
    top_k_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    run_dict: dict[str, dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        top = top_k_idx[qi]
        top_sorted = top[np.argsort(scores[qi, top])[::-1]]
        run_dict[str(qid)] = {str(corpus_ids[di]): float(scores[qi, di]) for di in top_sorted}
    qrels_str = {
        str(qid): {str(did): int(s) for did, s in rels.items()}
        for qid, rels in qrels.items()
    }
    ndcg = evaluate(Qrels(qrels_str), Run(run_dict), "ndcg@10", make_comparable=True)
    return task_name, float(ndcg)


# ── Main Evaluator ────────────────────────────────────────────────────────────

class LeaderboardEvaluator:
    def __init__(
        self,
        model_id: str,
        local_model_path: str,
        model_config: dict,
        lang_config: LanguageConfig,
        batch_sizes: dict[str, int],
        chroma_path: str = CHROMA_PATH,
        keep_chroma: bool = False,
        max_seq_length: int | None = None,
        results_suffix: str | None = None,
    ):
        self.model_id = model_id
        self.local_model_path = local_model_path
        self.model_config = model_config
        self.lang_config = lang_config
        self.batch_size = batch_sizes[model_config["tier"]]
        self.cache = EmbeddingCache(chroma_path)
        self.san_id = sanitize_model_id(model_id)
        self.results_dir = lang_config.results_base / self.san_id
        if results_suffix:
            new_base = lang_config.results_base.parent / f"{lang_config.results_base.name}{results_suffix}"
            new_base.mkdir(parents=True, exist_ok=True)
            self.results_dir = new_base / self.san_id
            self.results_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        self.keep_chroma = keep_chroma
        self.max_seq_length = max_seq_length

    def all_results_exist(self) -> bool:
        return all(
            (self.results_dir / f"{t}.json").exists()
            for t in self.lang_config.tasks
        )

    def run(self) -> dict[str, float]:
        if self.all_results_exist():
            logger.info("[%s] All %s results cached — skipping encode.", self.model_id, self.lang_config.lang_code.upper())
            return self._load_existing_results()

        t_wall = time.time()
        logger.info("[%s] Loading %d task datasets for %s…",
                    self.model_id, len(self.lang_config.tasks), self.lang_config.lang_code.upper())

        task_data_list = [_load_task_data(name, self.lang_config) for name in self.lang_config.tasks]

        logger.info("[%s] Loading model from %s (batch_size=%d)…",
                    self.model_id, self.local_model_path, self.batch_size)
        torch.cuda.reset_peak_memory_stats()

        # Pre-load patch: jina-embeddings-v5 registers JinaEmbeddingsV5Model in the
        # transformers auto registry on first load without config_class, which causes
        # auto_factory.from_pretrained to crash on any subsequent load of this model.
        import sys as _sys
        for _mod in list(_sys.modules.values()):
            _jina_cls = getattr(_mod, "__dict__", {}).get("JinaEmbeddingsV5Model")
            if _jina_cls is not None and not hasattr(_jina_cls, "config_class"):
                try:
                    _jina_cls.config_class = None
                except Exception:
                    pass

        model_kwargs: dict[str, Any] = dict(self.model_config.get("model_kwargs") or {})
        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])
        tokenizer_kwargs: dict[str, Any] = dict(self.model_config.get("tokenizer_kwargs") or {})
        model = SentenceTransformer(
            self.local_model_path,
            device="cuda",
            trust_remote_code=True,
            **({"model_kwargs": model_kwargs} if model_kwargs else {}),
            **({"tokenizer_kwargs": tokenizer_kwargs} if tokenizer_kwargs else {}),
        )
        model.eval()

        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
            logger.info("[Q3] Overrode max_seq_length to %d", self.max_seq_length)

        p_prefix = self.model_config.get("passage_prefix", "")
        q_prefix = self.model_config.get("query_prefix", "")
        c_encode_kwargs = self.model_config.get("corpus_encode_kwargs") or {}
        q_encode_kwargs = self.model_config.get("query_encode_kwargs") or {}

        results: dict[str, float] = {}
        total_corpus_hits = total_corpus_misses = 0
        all_query_encode_sec = 0.0

        # max_workers=1: ranx uses numba workqueue — crashes under concurrent threads
        executor = ThreadPoolExecutor(max_workers=1)

        with torch.amp.autocast("cuda"):
            for td in task_data_list:
                task_name = td["task_name"]

                # Resume: if this task's result was already written (e.g. from a prior
                # OOM attempt that completed some but not all tasks), skip re-encoding.
                result_file = self.results_dir / f"{task_name}.json"
                if result_file.exists():
                    cached = json.loads(result_file.read_text())
                    ndcg = cached["scores"]["test"][0]["ndcg_at_10"]
                    results[task_name] = ndcg
                    logger.info("[%s] %s: NDCG@10=%.4f  (resumed from file)",
                                self.model_id, task_name, ndcg)
                    continue

                t_task = time.time()

                c_ids = list(td["corpus"].keys())
                c_texts = [td["corpus"][i] for i in c_ids]
                q_ids = list(td["queries"].keys())
                q_texts = [td["queries"][i] for i in q_ids]

                c_embs, c_hits, c_misses = self.cache.cached_encode(
                    model, c_texts, c_ids,
                    _corpus_col(self.model_id, task_name, self.lang_config),
                    self.batch_size, p_prefix, c_encode_kwargs,
                )
                total_corpus_hits += c_hits
                total_corpus_misses += c_misses

                tq = time.time()
                q_embs, _, _ = self.cache.cached_encode(
                    model, q_texts, q_ids,
                    _query_col(self.model_id, task_name, self.lang_config),
                    self.batch_size, q_prefix, q_encode_kwargs,
                )
                all_query_encode_sec += time.time() - tq

                _, ndcg = executor.submit(
                    _compute_ndcg10, c_ids, c_embs, q_ids, q_embs, td["qrels"], task_name
                ).result()

                elapsed = time.time() - t_task
                results[task_name] = ndcg

                # Immediate per-task write
                self._write_task_result(task_name, ndcg, elapsed)
                logger.info("[%s] %s: NDCG@10=%.4f  (%.1fs)", self.model_id, task_name, ndcg, elapsed)

        executor.shutdown(wait=False)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        peak_vram = vram_peak_mb()
        total_sec = time.time() - t_wall

        log_json(
            self.results_dir / "_metadata.json",
            {
                "model_id": self.model_id,
                "lang_code": self.lang_config.lang_code,
                "peak_vram_mb": peak_vram,
                "query_encode_sec": round(all_query_encode_sec, 2),
                "total_wall_sec": round(total_sec, 2),
                "corpus_cache_hits": total_corpus_hits,
                "corpus_cache_misses": total_corpus_misses,
            },
        )

        avg = sum(results.values()) / len(results)
        logger.info(
            "[%s] %s=%.2f | wall=%.1fs | vram_peak=%dMB",
            self.model_id, self.lang_config.avg_label, avg * 100, total_sec, peak_vram,
        )

        if not self.keep_chroma:
            self._cleanup_chroma()

        return results

    def _write_task_result(self, task_name: str, ndcg: float, elapsed: float) -> None:
        log_json(
            self.results_dir / f"{task_name}.json",
            {
                "dataset_revision": "N/A",
                "evaluation_time": round(elapsed, 2),
                "scores": {"test": [{"ndcg_at_10": round(ndcg, 6), "main_score": round(ndcg, 6)}]},
            },
        )

    def _cleanup_chroma(self) -> None:
        deleted = 0
        for task_name in self.lang_config.tasks:
            for col_name in [
                _corpus_col(self.model_id, task_name, self.lang_config),
                _query_col(self.model_id, task_name, self.lang_config),
            ]:
                try:
                    self.cache.client.delete_collection(col_name)
                    deleted += 1
                except Exception:
                    pass
        logger.info("[%s] ChromaDB cleanup: deleted %d collections.", self.model_id, deleted)

    def _load_existing_results(self) -> dict[str, float]:
        out = {}
        for task in self.lang_config.tasks:
            data = json.loads((self.results_dir / f"{task}.json").read_text())
            out[task] = data["scores"]["test"][0]["main_score"]
        return out
