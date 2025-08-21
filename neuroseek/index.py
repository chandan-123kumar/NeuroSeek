import os
import numpy as np
from neuroseek.search import NeuroSeekSearch

class NeuroSeekIndex:
    def __init__(self):
        self.documents = {}   # {doc_id: vector}

    def add_document(self, doc_id: str, vector: np.ndarray, metadata: dict = None):
        """Insert a document vector into the index."""
        if doc_id in self.documents:
            raise ValueError(f"Document {doc_id} already exists!")
        self.documents[doc_id] = {
            "vector": vector,
            "metadata": metadata or {}
        }

    def search(self, query_vector: np.ndarray, top_k: int = 3):
        """Search for top-k similar documents."""
        results = []
        for doc_id, entry in self.documents.items():
            score = NeuroSeekSearch.cosine_similarity(query_vector, entry["vector"])
            results.append((doc_id, score, entry["metadata"]))
        
        # sort by similarity descending
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:top_k]

    
    def save(self, path: str = "neuroseek_index.npz"):
        """Save the index to disk."""
        doc_ids = list(self.documents.keys())
        vectors = [entry["vector"] for entry in self.documents.values()]
        metadata = [entry["metadata"] for entry in self.documents.values()]

        np.savez(path, doc_ids=doc_ids, vectors=vectors, metadata=metadata)
        print(f"[💾] Index saved at {path}")


    def load(self, path: str = "neuroseek_index.npz"):
        """Load the index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        data = np.load(path, allow_pickle=True)
        doc_ids = data["doc_ids"]
        vectors = data["vectors"]
        metadata = data["metadata"]

        self.documents = {
            doc_id: {"vector": vector, "metadata": meta}
            for doc_id, vector, meta in zip(doc_ids, vectors, metadata)
        }
        print(f"[📂] Index loaded from {path}, {len(self.documents)} docs")

    
