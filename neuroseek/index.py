import numpy as np
from neuroseek.search import NeuroSeekSearch

class NeuroSeekIndex:
    def __init__(self):
        self.documents = {}   # {doc_id: vector}

    def add_document(self, doc_id: str, vector: np.ndarray):
        """Insert a document vector into the index."""
        if doc_id in self.documents:
            raise ValueError(f"Document {doc_id} already exists!")
        self.documents[doc_id] = vector

    def search(self, query_vector: np.ndarray, top_k: int = 3):
        """Search for top-k similar documents."""
        results = []
        for doc_id, vector in self.documents.items():
            score = NeuroSeekSearch.cosine_similarity(query_vector, vector)
            results.append((doc_id, score))
        
        # sort by similarity descending
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:top_k]
