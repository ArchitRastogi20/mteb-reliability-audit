"""
S4: Multi-hop vs summarization breakdown for §5.1.

Verify R3's surfaced finding: small models hold up better on multi-hop than on
summarization (contradicts conventional practitioner intuition).

For each (small_model, large_model) pair × (config, query_type):
  ratio = small_NDCG@10 / large_NDCG@10

Headline: jina-v5-nano (212M) vs Qwen3-8B across multi-hop vs summarization.
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(REPO_ROOT / 'analysis/analysis_output/unified_rag_bytype_results_v2.csv')

SMALL = 'jina-v5-nano'
LARGE = 'Qwen3-Embedding-8B'  # check exact short_name first

print("Unique short_names containing 'qwen' or 'jina' or '8b':",
      [n for n in df['short_name'].unique() if any(k in n.lower() for k in ['qwen','jina','8b'])])
print("\nUnique query_types:", df['query_type'].unique())
print("Unique datasets:", df['dataset'].unique())

# Determine actual large-model short_name
candidates = [n for n in df['short_name'].unique() if 'qwen' in n.lower() and '8' in n.lower()]
print(f"\nQwen-8B candidate(s): {candidates}")
LARGE = candidates[0]

# Build small/large pivot
sub = df[df['short_name'].isin([SMALL, LARGE])][['short_name', 'dataset', 'query_type', 'ndcg_at_10']]
pivot = sub.pivot_table(index=['dataset', 'query_type'], columns='short_name',
                       values='ndcg_at_10').reset_index()
pivot['ratio_pct'] = (pivot[SMALL] / pivot[LARGE]) * 100

print("\nPer-(dataset, query_type) ratio (jina-v5-nano / Qwen3-8B, %):")
print(pivot.to_string(index=False))

# Group by query type
print("\nBy query type — ratio range and median across the 6 datasets:")
qt_summary = pivot.groupby('query_type')['ratio_pct'].agg(['min','median','max','count'])
print(qt_summary.to_string())

# Save results
results = {
    'small_model': SMALL,
    'large_model': LARGE,
    'per_config_query_type': pivot.to_dict(orient='records'),
    'by_query_type_summary': qt_summary.to_dict(orient='index'),
}

# Quick headline numbers
mh = pivot[pivot['query_type'] == 'multi_hop']
summ = pivot[pivot['query_type'] == 'summarization']
fact = pivot[pivot['query_type'] == 'factual']

print("\n" + "="*72)
print("HEADLINE NUMBERS for §5.1 (small=jina-v5-nano, large=Qwen3-8B):")
print("="*72)
if len(mh):
    print(f"  Multi-hop:      {mh['ratio_pct'].min():.0f}–{mh['ratio_pct'].max():.0f}% "
          f"(median {mh['ratio_pct'].median():.0f}%) across {len(mh)} configs")
if len(fact):
    print(f"  Factual:        {fact['ratio_pct'].min():.0f}–{fact['ratio_pct'].max():.0f}% "
          f"(median {fact['ratio_pct'].median():.0f}%)")
if len(summ):
    print(f"  Summarization:  {summ['ratio_pct'].min():.0f}–{summ['ratio_pct'].max():.0f}% "
          f"(median {summ['ratio_pct'].median():.0f}%)")

with open(REPO_ROOT / 'analysis/analysis_output/stats/rev_S4_query_type_ratios.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved to: analysis_output/stats/rev_S4_query_type_ratios.json")
