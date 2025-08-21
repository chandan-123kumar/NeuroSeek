import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from neuroseek.index import NeuroSeekIndex

# Step 1: Create index and add documents with metadata
index = NeuroSeekIndex()
index.add_document(
    "doc1",
    np.array([0.1, 0.2, 0.3]),
    metadata={"title": "Intro to AI", "tags": ["AI", "Basics"]}
)
index.add_document(
    "doc2",
    np.array([0.4, 0.5, 0.6]),
    metadata={"title": "Deep Learning Guide", "tags": ["DL", "NeuralNets"]}
)

# Step 2: Save to disk
index.save("my_index_meta.npz")

# Step 3: Load back into a new instance
new_index = NeuroSeekIndex()
new_index.load("my_index_meta.npz")

# Step 4: Run search on loaded index
query = np.array([0.2, 0.25, 0.3])
results = new_index.search(query, top_k=2)

print("Search Results with Metadata:")
for doc_id, score, meta in results:
    print(f"{doc_id} | {meta.get('title')} | Score={score:.4f}")
