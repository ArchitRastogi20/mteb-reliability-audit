import sys
import os

# Make rag-dataset's `scripts` package importable as `scripts._openai_embed_lib`
# etc. rag-dataset/ is a sibling of analysis/, not a child.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag-dataset"))
