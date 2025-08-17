import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from neuroseek.index import NeuroSeekIndex


# Initialize index
index = NeuroSeekIndex()

# Add documents (3 random vectors for demo)
index.add_document("doc1", np.array([0.1, 0.2, 0.3]))
index.add_document("doc2", np.array([0.2, 0.1, 0.4]))
index.add_document("doc3", np.array([0.9, 0.8, 0.7]))

# Query vector
query = np.array([0.15, 0.2, 0.25])

# Search
results = index.search(query, top_k=3)

print("Top Results:")
for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
