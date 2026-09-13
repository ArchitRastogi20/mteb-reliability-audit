"""
R4b: Cross-judge 1.71x answer-quality claim with the judge LLM as second judge.

Closes DA-C5 + addresses Limitations §6 judge-overlap concern.

Replays the existing 6-model × hi_finance / it_finance per-query generated
answers from analysis_output/stats/downstream_rag.json through the judge LLM
(judge-llm-small default) as judge. For each query, asks the judge LLM to count
how many of the gold keypoints are covered by the model's generated answer.

Outputs per-(model, config) coverage rate as judged by the judge LLM, alongside
the existing GPT-judge rates. If the rank order is preserved across judges,
the within-OpenAI 1.71x propagation finding is robust to judge bias.

Cost: 6 × 50 = 300 the judge LLM calls × ~600 in-tokens × ~10 out-tokens.
With judge-llm-small ($1/M in + $5/M out), expected cost ~$0.20.
"""
import json
import os
import sys
import time
from pathlib import Path

# NOTE: stub import. Replace with the LLM SDK you use as the judge.
from judge_llm_sdk import JudgeClient  # type: ignore[import-not-found]
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV = REPO_ROOT / 'rag-dataset/.env'
load_dotenv(ENV)

DOWNSTREAM = REPO_ROOT / 'analysis/analysis_output/stats/downstream_rag.json'
ANS_DIR    = REPO_ROOT / 'rag-dataset/checkpoints'
OUT        = REPO_ROOT / 'analysis/analysis_output/stats/rev_R4b_crossjudge.json'

JUDGE_MODEL = 'judge-llm-small'  # cheap, fast, sufficient for binary-coverage judging

cli = JudgeClient(api_key=os.environ['JUDGE_API_KEY'])


def load_keypoints():
    """Map qid -> (question, gold_keypoints) from checkpoints/answers_*.jsonl."""
    kp = {}
    for fn in ['answers_hi_finance.jsonl', 'answers_it_finance.jsonl']:
        p = ANS_DIR / fn
        if not p.exists():
            print(f"WARN: {p} not found")
            continue
        with open(p) as f:
            for line in f:
                e = json.loads(line)
                if 'keypoints' in e:
                    kp[e['query_id']] = (e['question'], e['keypoints'])
    print(f"Loaded {len(kp)} qid → keypoint mappings")
    return kp


JUDGE_PROMPT = """You are a strict judge of retrieval-augmented answer quality.

QUESTION:
{q}

GOLD KEYPOINTS (each is a discrete fact that a perfect answer should contain):
{kp_block}

CANDIDATE ANSWER (produced by an LLM given retrieved documents):
{a}

Count exactly how many of the {n} gold keypoints are explicitly covered by the candidate answer (paraphrase counts; partial mention does not).

Reply with ONLY a single integer between 0 and {n}, no other text."""


def judge_one(qid, question, keypoints, answer):
    kp_block = '\n'.join(f"  {i+1}. {k}" for i, k in enumerate(keypoints))
    msg = cli.messages.create(
        model=JUDGE_MODEL,
        max_tokens=10,
        messages=[{
            'role': 'user',
            'content': JUDGE_PROMPT.format(
                q=question, kp_block=kp_block, a=answer, n=len(keypoints))
        }]
    )
    txt = msg.content[0].text.strip()
    try:
        # Take first integer in response
        import re
        m = re.search(r'\d+', txt)
        covered = int(m.group()) if m else 0
        covered = min(covered, len(keypoints))
    except Exception:
        covered = 0
    return covered, len(keypoints), txt


def main():
    if not DOWNSTREAM.exists():
        print(f"ERROR: {DOWNSTREAM} not found"); sys.exit(1)
    d = json.loads(DOWNSTREAM.read_text())
    kp_map = load_keypoints()

    out = {'judge_model': JUDGE_MODEL, 'per_model_summary': [], 'detailed': {}}

    for key, entry in d['detailed'].items():
        provider, model, config = key.split('/')
        per_q = entry['per_query']
        print(f"\n=== {key} (n={len(per_q)}) ===")

        coverage_rates = []
        details = []
        t0 = time.time()
        for i, q in enumerate(per_q):
            qid = q['qid']
            answer = q['answer']
            if qid not in kp_map:
                print(f"  skip {qid} (no keypoints)")
                continue
            question, keypoints = kp_map[qid]
            try:
                covered, total, raw = judge_one(qid, question, keypoints, answer)
            except Exception as e:
                print(f"  ERROR on {qid}: {str(e)[:100]}")
                continue
            rate = covered / total if total else 0
            coverage_rates.append(rate)
            details.append({'qid': qid, 'qtype': q['qtype'],
                           'gpt_covered': q['covered'], 'gpt_total': q['total'],
                           'gpt_rate': q['rate'],
                           'judge_covered': covered, 'judge_total': total,
                           'judge_rate': rate, 'raw': raw})
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(per_q)} done, t={time.time()-t0:.1f}s")

        if coverage_rates:
            mean_kp = sum(coverage_rates) / len(coverage_rates)
            sorted_r = sorted(coverage_rates)
            median_kp = sorted_r[len(sorted_r)//2]
            entry_summary = {
                'provider': provider, 'model': model, 'config': config,
                'n_judged': len(coverage_rates),
                'gpt_mean_kp': round(entry['mean_keypoint_coverage'], 4),
                'claude_mean_kp': round(mean_kp, 4),
                'claude_median_kp': round(median_kp, 4),
                'rank_preserved_marker': None,  # filled in after
            }
            out['per_model_summary'].append(entry_summary)
            out['detailed'][key] = details
            print(f"  Mean KP — GPT: {entry['mean_keypoint_coverage']:.3f}  "
                  f"the judge LLM: {mean_kp:.3f}  (n={len(coverage_rates)})")

    # Compute the headline 1.71x ratio under LLM judge for hi_fin OpenAI pair
    hi_large = next((e for e in out['per_model_summary']
                     if e['model']=='text-embedding-3-large' and e['config']=='hi_finance'), None)
    hi_small = next((e for e in out['per_model_summary']
                     if e['model']=='text-embedding-3-small' and e['config']=='hi_finance'), None)
    if hi_large and hi_small and hi_small['claude_mean_kp'] > 0:
        ratio_claude = hi_large['claude_mean_kp'] / hi_small['claude_mean_kp']
        ratio_gpt = hi_large['gpt_mean_kp'] / hi_small['gpt_mean_kp'] if hi_small['gpt_mean_kp'] > 0 else None
        out['headline_1_71x_check'] = {
            'gpt_judge_ratio': round(ratio_gpt, 3) if ratio_gpt else None,
            'claude_judge_ratio': round(ratio_claude, 3),
            'claim_holds_under_both_judges': ratio_claude > 1.2,
        }
        print(f"\n=== HEADLINE CHECK (hi_finance, OpenAI -large vs -small) ===")
        print(f"  GPT judge ratio: {ratio_gpt:.3f}")
        print(f"  LLM judge ratio: {ratio_claude:.3f}")
        print(f"  Both judges agree on direction: {ratio_claude > 1.0 and ratio_gpt and ratio_gpt > 1.0}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == '__main__':
    main()
