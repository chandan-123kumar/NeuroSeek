
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroseek.storage import NeuroSeekStorage
from neuroseek.search import NeuroSeekSearch

# 1. Initialize storage and search
db = NeuroSeekStorage()
search_engine = NeuroSeekSearch(db)

# 2. Insert sample data
db.insert("doc1", {"title": "AI in Healthcare"}, [0.12, 0.45, 0.67, 0.89])
db.insert("doc2", {"title": "Climate Change Research"}, [0.91, 0.34, 0.55, 0.12])
db.insert("doc3", {"title": "AI for Climate Solutions"}, [0.50, 0.40, 0.60, 0.80])

# 3. Perform a search
query_vector = [0.10, 0.50, 0.65, 0.85]
results = search_engine.search(query_vector, top_k=2)

print("\n[🔍] Search Results for query_vector:")
for oid, score in results:
    print(f"{oid} | Score: {score:.4f} | Title: {db.get(oid)['metadata']['title']}")
