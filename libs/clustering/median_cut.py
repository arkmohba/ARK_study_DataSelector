import numpy as np
from typing import List
import math

from .vector_wrapper import VectorWrapper


class MedianCut:
    """Vectorの配列を均等に分割するMedianCutクラス
    """

    def __init__(self):
        self.clusters: List[List[VectorWrapper]] = []

    @staticmethod
    def calc_n_devide(n_data, max_element_size):
        """均等に2分割してmaxの個数まで減る回数を計算する
        n_data / 2^n  < max_element_size
        log_2(n_data / max_element_size) < n


        Args:
            max_element_size ([type]): [description]
            n_data ([type]): [description]

        Returns:
            [type]: [description]
        """
        # n_data / 2^n  < max_element_size
        # log_2(n_data / max_element_size) < n
        n_devide = math.log2(n_data / max_element_size)
        n_devide = math.ceil(n_devide)  # 大きい方に整数化する
        return n_devide

    def divide_2group(self, vectors: List[VectorWrapper]):
        """データをmin, max差が一番大きい軸の方向に2分割する。

        Args:
            vectors (List[VectorWrapper]): [description]

        Returns:
            [type]: [description]
        """
        data = [vector.vector for vector in vectors]
        # min, maxの差が最大になる軸を調査
        data = np.array(data)  # [N=n_data, k=vector_size]
        data_diff = data.max(axis=0) - data.min(axis=0)  # [k]
        target_array = data[:, data_diff.argmax()]  # [N]
        target_indices = np.argsort(target_array).tolist()

        # 選択軸で並び替えをして中心で分割
        half_point = len(target_indices) // 2
        new_vectors_0 = [vectors[i] for i in target_indices[:half_point]]
        new_vectors_1 = [vectors[i] for i in target_indices[half_point:]]
        return new_vectors_0, new_vectors_1

    def divide(
            self,
            vectors: List[VectorWrapper],
            max_element_size: int) -> List[List[VectorWrapper]]:
        """データをmax_element_sizeになるまで分割する

        Args:
            vectors (List[VectorWrapper]): ベクトルの配列
            max_element_size (int): クラスタの最大要素数

        Returns:
            List[List[VectorWrapper]]: クラスタに分割したベクトルの配列（clusters[cluster_id][vector_id]）
        """
        # 分割回数
        n_devide = MedianCut.calc_n_devide(len(vectors), max_element_size)

        # 初期分割
        self.clusters = [vectors]

        for _ in range(n_devide):
            new_clusters = []
            for cluster in self.clusters:
                new_cluster_1, new_cluster_2 = self.divide_2group(cluster)
                new_clusters.extend([new_cluster_1, new_cluster_2])
            self.clusters = new_clusters
        return self.clusters
