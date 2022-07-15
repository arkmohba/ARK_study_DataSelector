from __future__ import annotations

import numpy as np
from typing import List
import matplotlib.pyplot as plt


class VectorWrapper:
    """Vectorの配列のラッパー
    """

    def __init__(self, vector: np.ndarray, id: int = -1):
        self.id = id
        self.vector: np.ndarray = vector
        self.weight: float = 1
        self.cluster_indices: list = []
        self.parent_id: int = -1

    def dist(self, another: VectorWrapper):
        return np.linalg.norm((self.vector - another.vector))

    @staticmethod
    def plot_vectors(vectors: List[VectorWrapper], color_as_parent: bool = False, **dist):
        if len(vectors):
            data = np.array([vec.vector for vec in vectors]).T
            circle_sizes = np.array([vec.weight for vec in vectors])
            if color_as_parent:
                colors = [vec.parent_id for vec in vectors]
                edgecolors = None
            else:
                colors = colors = [vec.id for vec in vectors]
                edgecolors = "red"

            plt.scatter(data[0], data[1],
                        s=circle_sizes * 20,
                        c=colors, edgecolors=edgecolors,
                        **dist)

    @staticmethod
    def create_vector_list(vectors_list: List[np.ndarray]):
        vectors = [VectorWrapper(vector, i) for i, vector in enumerate(vectors_list)]
        return vectors
