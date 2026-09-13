"""
S1: API embedding deployment numbers — measure client-side latency + pull
public pricing for OpenAI, Cohere, Gemini embedding endpoints.

Output: analysis_output/stats/rev_S1_api_deployment.json with:
- per-provider mean p50/p99 latency over N=50 single-query embedding calls
- public $/1M-token pricing
- rate-limit notes
"""
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = REPO_ROOT / 'rag-dataset/.env'
load_dotenv(ENV_FILE)

OUT = REPO_ROOT / 'analysis/analysis_output/stats/rev_S1_api_deployment.json'

# Sample queries: short Hindi finance questions (representative of RAG workload)
SAMPLE = [
    "What is the EPS of company X for FY2023?",
    "Quale è la crescita ricavi nel 2024?",
    "三菱UFJ銀行の純利益は何円ですか?",
    "एनएसई का बाजार पूंजीकरण क्या है?",
    "What was the dividend declared in Q3?",
] * 10  # 50 queries total


def stats(latencies):
    s = sorted(latencies)
    return {
        'n': len(s),
        'mean_ms': round(statistics.mean(s), 1),
        'p50_ms': round(s[len(s)//2], 1),
        'p99_ms': round(s[min(len(s)-1, int(len(s)*0.99))], 1),
        'min_ms': round(s[0], 1),
        'max_ms': round(s[-1], 1),
    }


def measure_openai(model='text-embedding-3-small'):
    from openai import OpenAI
    cli = OpenAI(api_key=os.environ['CHATGPT_API_KEY'])
    lat = []
    for q in SAMPLE:
        t0 = time.perf_counter()
        cli.embeddings.create(model=model, input=q)
        lat.append((time.perf_counter() - t0) * 1000)
    return stats(lat)


def measure_cohere(model='embed-multilingual-v3.0'):
    import cohere
    cli = cohere.ClientV2(api_key=os.environ['COHERE_API_KEY'])
    lat = []
    for q in SAMPLE:
        t0 = time.perf_counter()
        cli.embed(texts=[q], model=model, input_type='search_query',
                  embedding_types=['float'])
        lat.append((time.perf_counter() - t0) * 1000)
    return stats(lat)


def measure_gemini(model='models/gemini-embedding-2'):
    from google import genai
    cli = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
    lat = []
    for q in SAMPLE:
        t0 = time.perf_counter()
        cli.models.embed_content(model=model, contents=q)
        lat.append((time.perf_counter() - t0) * 1000)
    return stats(lat)


PRICING = {
    'openai/text-embedding-3-small': {'usd_per_1M_tokens': 0.020, 'dim': 1536, 'tier': 'API'},
    'openai/text-embedding-3-large': {'usd_per_1M_tokens': 0.130, 'dim': 3072, 'tier': 'API'},
    'cohere/embed-multilingual-v3.0': {'usd_per_1M_tokens': 0.100, 'dim': 1024, 'tier': 'API',
                                       'notes': 'Free tier: 1000 calls/month; production: $0.10/1M tokens'},
    'gemini/gemini-embedding-2':     {'usd_per_1M_tokens': 0.150, 'dim': 3072, 'tier': 'API',
                                       'notes': 'Same model used in §4.5/Table 4 evaluation'},
}


def main():
    results = {'pricing': PRICING, 'latency': {}, 'sample_size': len(SAMPLE),
               'measured_from': 'WSL2 client (single region, 1-by-1 sequential calls)'}

    print(f"Measuring API embedding latency over n={len(SAMPLE)} single-query calls...")
    for name, fn in [
        ('openai/text-embedding-3-small', lambda: measure_openai('text-embedding-3-small')),
        ('openai/text-embedding-3-large', lambda: measure_openai('text-embedding-3-large')),
        ('cohere/embed-multilingual-v3.0', measure_cohere),
        ('gemini/gemini-embedding-2', measure_gemini),
    ]:
        try:
            print(f"  {name} ...", end=' ', flush=True)
            r = fn()
            results['latency'][name] = r
            print(f"p50={r['p50_ms']}ms p99={r['p99_ms']}ms")
        except Exception as e:
            results['latency'][name] = {'error': str(e)[:200]}
            print(f"ERROR: {str(e)[:120]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT}")
    print("\nSummary table for paper §5.2:")
    print(f"  {'Provider/Model':<40} {'$/1M':>8} {'p50':>8} {'p99':>8}")
    print("  " + "-"*68)
    for k in PRICING:
        price = PRICING[k]['usd_per_1M_tokens']
        lat = results['latency'].get(k, {})
        p50 = lat.get('p50_ms', 'err')
        p99 = lat.get('p99_ms', 'err')
        print(f"  {k:<40} ${price:>6.3f} {str(p50):>8} {str(p99):>8}")


if __name__ == '__main__':
    main()
