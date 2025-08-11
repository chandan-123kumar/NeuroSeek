import numpy as np
from typing import Any, List, Dict

class NeuroSeekStorage:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def insert(self, object_id: str, metadata: Dict[str, Any], vector: List[float]):
        """
        Insert an object into the storage.
        """
        if object_id in self.store:
            raise ValueError(f"Object with ID '{object_id}' already exists.")
        self.store[object_id] = {
            "metadata": metadata,
            "vector": np.array(vector, dtype=np.float32)
        }
        print(f"[✅] Inserted object: {object_id}")

    def get(self, object_id: str) -> Dict[str, Any]:
        """
        Retrieve object by ID.
        """
        return self.store.get(object_id, None)

    def all_objects(self):
        """
        Return all stored objects.
        """
        return self.store
