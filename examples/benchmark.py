import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from neuroseek.index import NeuroSeekIndex

class Benchmark:
    @staticmethod
    def run(storage: NeuroSeekIndex, query_vector: np.ndarray, top_k: int = 3, trials: int = 10):
        times = []
        for _ in range(trials):
            start = time.time()
            result = storage.search(query_vector, top_k=top_k)
            print(f"Query Result: {result}")
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        print(f"Average Query Time over {trials} runs: {avg_time:.6f} seconds")
        return avg_time

if __name__ == "__main__":
    # Example Benchmark Run
    dim = 128
    storage = NeuroSeekIndex()
    np.random.seed(42)

    # Insert 10,000 random vectors
    for i in range(10000):
        vec = np.random.rand(dim).astype(np.float32)
        metadata = {"title": f"Item {i}"}
        storage.add_document(f"id_{i}", vec, metadata)

    # Run benchmark
    query_vec = np.random.rand(dim).astype(np.float32)
    Benchmark.run(storage, query_vec, top_k=5)
