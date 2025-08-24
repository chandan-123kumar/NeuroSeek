# neuroseek/ann_index.py
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple

class ANNIndex:
    def __init__(self, n_clusters: int = 5):
        """
        ANN Index using KMeans clustering (simplified IVF).
        n_clusters: how many clusters to partition vectors into
        """
        self.n_clusters = n_clusters
        self.kmeans = None
        self.vectors = []
        self.ids = []
        self.cluster_assignments = None

    def build(self, vectors: List[np.ndarray], ids: List[str]):
        """
        Build the ANN index from given vectors.
        """
        self.vectors = np.array(vectors)
        self.ids = ids
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_assignments = self.kmeans.fit_predict(self.vectors)

    def search(self, query: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Approximate search:
        1. Find the nearest cluster
        2. Search only inside that cluster
        """
        cluster = self.kmeans.predict(query.reshape(1, -1))[0]

        cluster_indices = np.where(self.cluster_assignments == cluster)[0]
        cluster_vectors = self.vectors[cluster_indices]
        cluster_ids = [self.ids[i] for i in cluster_indices]

        distances = np.linalg.norm(cluster_vectors - query, axis=1)
        sorted_idx = np.argsort(distances)[:top_k]

        return [(cluster_ids[i], float(distances[i])) for i in sorted_idx]
