import numpy as np
from typing import List, Tuple
from .storage import NeuroSeekStorage

class NeuroSeekSearch:
    def __init__(self, storage: NeuroSeekStorage):
        self.storage = storage

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        dot = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Brute-force search: compare query_vector with all stored vectors.
        Returns top_k results sorted by similarity.
        """
        results = []
        query_vec = np.array(query_vector, dtype=np.float32)

        for object_id, obj in self.storage.all_objects().items():
            similarity = self.cosine_similarity(query_vec, obj["vector"])
            print(object_id)
            results.append((object_id, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
