"""Closed-book baseline: answer each RAG query WITHOUT retrieved context,
score with two judges (OpenAI mini + Claude haiku), aggregate keypoint
coverage per config, and write a comparison table vs BM25 / best-dense.

Hard cost ceiling: $10. SQLite cache prevents re-spend on re-runs.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sqlite3, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

sys.path.insert(0, str(REPO))
from _common import RAG_OUTPUT, RAG_EVAL, RAG_CONFIGS, OUTPUTS, OUTPUTS

ANSWER_MODEL  = "gpt-5.4-mini"
JUDGE_OPENAI  = "gpt-5.4-mini"
JUDGE_CLAUDE  = "claude-haiku-4-5"

# Prices in USD per 1K tokens (input, output)
PRICE = {
    "gpt-5.4-mini":      (0.00075, 0.0045),
    "claude-haiku-4-5":  (0.0010,  0.0050),
}

CEILING_USD = 10.00

JUDGE_PROMPT = (
    "You are a strict judge of answer quality.\n\n"
    "QUESTION:\n{q}\n\n"
    "GOLD KEYPOINTS (each is a discrete fact the answer should contain):\n"
    "{kp_block}\n\n"
    "CANDIDATE ANSWER:\n{a}\n\n"
    "Count how many of the {n} gold keypoints are explicitly covered "
    "by the candidate answer (paraphrase counts; partial mention does not). "
    "Reply with ONLY a single integer between 0 and {n}, no other text."
)

ANSWER_SYS = (
    "Answer the user's question concisely using only your own knowledge. "
    "Reply in the same language as the question."
)


def prompt_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheStore:
    path: Path

    def __post_init__(self):
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS cache (
            config TEXT, query_id TEXT, model TEXT, prompt_hash TEXT,
            content TEXT, cost_usd REAL,
            PRIMARY KEY (config, query_id, model, prompt_hash))""")
        self.conn.commit()

    def get(self, config, qid, model, h) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT content FROM cache WHERE config=? AND query_id=? AND model=? AND prompt_hash=?",
            (config, qid, model, h))
        r = cur.fetchone()
        return r[0] if r else None

    def put(self, config, qid, model, h, content, cost_usd):
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
            (config, qid, model, h, content, cost_usd))
        self.conn.commit()

    def total_cost_usd(self) -> float:
        cur = self.conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM cache")
        return float(cur.fetchone()[0])


def _cost(model: str, in_tokens: int, out_tokens: int) -> float:
    pin, pout = PRICE[model]
    return (in_tokens / 1000.0) * pin + (out_tokens / 1000.0) * pout


def _call_openai(client, model: str, system: str, user: str, max_tokens: int) -> tuple[str, float]:
    kwargs = dict(
        model=model, max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
    )
    try:
        resp = client.chat.completions.create(temperature=0, **kwargs)
    except Exception as e:
        # Some newer reasoning-style models reject `temperature`; retry without it.
        if "temperature" in str(e).lower():
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    usage = resp.usage
    cost = _cost(model, usage.prompt_tokens, usage.completion_tokens)
    return resp.choices[0].message.content.strip(), cost


def _call_claude(client, model: str, prompt: str, max_tokens: int) -> tuple[str, float]:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = resp.usage
    cost = _cost(model, usage.input_tokens, usage.output_tokens)
    return resp.content[0].text.strip(), cost


def _load_queries(lang: str, domain: str) -> list[dict]:
    p = RAG_OUTPUT / lang / domain / "full_queries.jsonl"
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("query_type") == "unanswerable":
                continue
            kp = (d.get("ground_truth") or {}).get("keypoints") or []
            if not kp:
                continue
            out.append({"qid": d["_id"], "question": d["question"], "keypoints": kp})
    return out


def _load_dense_ndcg_at_10(lang: str, domain: str) -> dict[str, float]:
    """Per-model overall NDCG@10 for one config, from existing eval JSONs.

    Eval JSON schema: {model_id, lang, domain, overall: {ndcg_at_10, ...},
                       by_type: {factual: {...}, multi_hop: {...}, ...}}
    """
    out: dict[str, float] = {}
    if not RAG_EVAL.exists():
        return out
    for model_dir in RAG_EVAL.iterdir():
        f = model_dir / f"{lang}_{domain}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        overall = d.get("overall") or {}
        n = overall.get("ndcg_at_10")
        if n is not None:
            out[model_dir.name] = float(n)
    return out


def _parse_int(txt: str, n: int) -> int:
    m = re.search(r"\d+", txt)
    if not m:
        return 0
    return min(int(m.group()), n)


def run_config(lang: str, domain: str, cache: CacheStore,
               openai_client, claude_client, dry_run: bool,
               openai_only: bool = False) -> dict:
    config = f"{lang}_{domain}"
    queries = _load_queries(lang, domain)
    if dry_run:
        queries = queries[:1]
    n_q = len(queries)
    covered_openai = 0; covered_claude = 0; total_kp = 0
    n_ans_calls = n_jo_calls = n_jc_calls = 0
    t0 = time.time()
    for q in queries:
        if cache.total_cost_usd() > CEILING_USD:
            print(f"[abort] hit ${CEILING_USD} ceiling; partial results for {config}")
            break
        ph_ans = prompt_hash(ANSWER_SYS + "||" + q["question"])
        ans = cache.get(config, q["qid"], ANSWER_MODEL, ph_ans)
        if ans is None:
            ans, cost = _call_openai(openai_client, ANSWER_MODEL,
                                     ANSWER_SYS, q["question"], 400)
            cache.put(config, q["qid"], ANSWER_MODEL, ph_ans, ans, cost)
            n_ans_calls += 1

        kp_block = "\n".join(f"  {i+1}. {k}" for i, k in enumerate(q["keypoints"]))
        n = len(q["keypoints"])
        judge_user = JUDGE_PROMPT.format(q=q["question"], kp_block=kp_block,
                                          a=ans, n=n)

        ph_jo = prompt_hash("openai-judge||" + judge_user)
        oj = cache.get(config, q["qid"], JUDGE_OPENAI, ph_jo)
        if oj is None:
            oj, cost = _call_openai(openai_client, JUDGE_OPENAI,
                                    "You are a strict numeric judge.", judge_user, 16)
            cache.put(config, q["qid"], JUDGE_OPENAI, ph_jo, oj, cost)
            n_jo_calls += 1
        covered_openai += _parse_int(oj, n)

        if not openai_only:
            ph_jc = prompt_hash("claude-judge||" + judge_user)
            cj = cache.get(config, q["qid"], JUDGE_CLAUDE, ph_jc)
            if cj is None:
                cj, cost = _call_claude(claude_client, JUDGE_CLAUDE, judge_user, 16)
                cache.put(config, q["qid"], JUDGE_CLAUDE, ph_jc, cj, cost)
                n_jc_calls += 1
            covered_claude += _parse_int(cj, n)

        total_kp += n

    dense_ndcg = _load_dense_ndcg_at_10(lang, domain)
    best_dense_model, best_dense_ndcg = (None, float("nan"))
    if dense_ndcg:
        # exclude bm25 when picking "best dense"
        dense_only = {k: v for k, v in dense_ndcg.items() if k != "bm25"}
        if dense_only:
            best_dense_model, best_dense_ndcg = max(dense_only.items(),
                                                   key=lambda kv: kv[1])
    bm25_ndcg = dense_ndcg.get("bm25", float("nan"))

    return {
        "config": config,
        "n_queries_scored": n_q if total_kp else 0,
        "total_keypoints": total_kp,
        "coverage_openai_judge": (covered_openai / total_kp) if total_kp else float("nan"),
        "coverage_claude_judge": ((covered_claude / total_kp) if total_kp else float("nan"))
                                  if not openai_only else None,
        "bm25_ndcg_at_10": bm25_ndcg,
        "best_dense_model": best_dense_model,
        "best_dense_ndcg_at_10": best_dense_ndcg,
        "wall_seconds": round(time.time() - t0, 1),
        "new_api_calls": n_ans_calls + n_jo_calls + n_jc_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="1 query per config; spends pennies")
    parser.add_argument("--openai-only", action="store_true",
                        help="skip the Claude judge (use when Claude quota is exhausted)")
    args = parser.parse_args()

    from openai import OpenAI

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    claude_client = None
    if not args.openai_only:
        from anthropic import Anthropic
        claude_client = Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

    cache_path = OUTPUTS / "exp11_closed_book_cache.sqlite"
    cache = CacheStore(cache_path)
    print(f"prior cache cost: ${cache.total_cost_usd():.4f}")

    rows = []
    for lang, domain in RAG_CONFIGS:
        print(f"\n=== {lang}_{domain} ===")
        r = run_config(lang, domain, cache, openai_client, claude_client,
                       args.dry_run, openai_only=args.openai_only)
        rows.append(r)
        print(json.dumps(r, indent=2))
        if cache.total_cost_usd() >= CEILING_USD:
            print(f"[abort] reached ${CEILING_USD}; remaining configs skipped")
            break

    df = pd.DataFrame(rows)
    out_csv = OUTPUTS / "exp11_rag_with_closedbook.csv"
    df.to_csv(out_csv, index=False)
    out_json = OUTPUTS / "exp11_closed_book_summary.json"
    out_json.write_text(json.dumps({"rows": rows,
                                    "total_cost_usd": cache.total_cost_usd()},
                                    indent=2, default=str))
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_json}")
    print(f"total cost so far: ${cache.total_cost_usd():.4f}")


if __name__ == "__main__":
    main()
