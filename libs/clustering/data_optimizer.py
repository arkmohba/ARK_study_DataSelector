from typing import List
from tqdm import tqdm
import random
import math
from sklearn.metrics import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
import joblib
import numpy as np

from .median_cut import MedianCut
from .annealing import AnnealingSampler
from .vector_wrapper import VectorWrapper


def do_annealing(cluster, near_limit, n_sample, use_dwave, use_amplify):
    sampler = AnnealingSampler(
        n_sample=n_sample, use_dwave=use_dwave, use_amplify=use_amplify)
    return sampler.select(cluster, near_limit)


class DataOptimizer:
    def __init__(self,
                 max_data_size: int = 100,
                 n_sample: int = 100,
                 use_dwave: bool = False,
                 use_amplify=False):
        self.median_cut: MedianCut = MedianCut()
        self.n_sample = n_sample
        self.use_dwave = use_dwave
        self.use_amplify = use_amplify
        self.sampler: AnnealingSampler = AnnealingSampler(
            n_sample=n_sample, use_dwave=use_dwave, use_amplify=use_amplify)
        self.max_data_size: int = max_data_size

    def select(self, vectors: List[VectorWrapper], minimum_limit: int = 10):
        all_leaved = vectors
        all_dropped = []  # dropしたvectorの受け皿
        iter_size = 10
        print("selecting...")
        for iter in tqdm(range(iter_size)):
            clusters: List[List[VectorWrapper]] = self.median_cut.divide(
                all_leaved, self.max_data_size)

            # QUBO作成時に計算した距離情報を使う
            self.sampler.setup_qubo(clusters[0], 0)
            # 距離行列の対角を除いて値を計算
            target_array = self.sampler.distance_mat[~np.diag(
                [True] * len(clusters[0]))]
            near_limit = (target_array.min() +
                          target_array.mean()) / 2  # 最小値と平均値の平均値

            results = []
            connected_indices_list = []
            n_leave = 0  # 残った個数
            cluster: List[VectorWrapper]
            # 並列化版
            parallel_outputs = joblib.Parallel(n_jobs=-1)(joblib.delayed(do_annealing)(
                cluster, near_limit, self.n_sample, self.use_dwave, self.use_amplify) for cluster in clusters)
            # 直列化版
            # parallel_outputs = [do_annealing(cluster, near_limit, self.n_sample, self.use_dwave, self.use_amplify) for cluster in clusters]
            for will_leave, connected_indices in parallel_outputs:
                results.append(will_leave)
                connected_indices_list.append(connected_indices)
                n_leave += np.count_nonzero(will_leave)

            if iter != 0 and n_leave < minimum_limit:
                # 下限を下回った場合は一つ前の結果を返す（初回を除く）
                print("len(all_leaved_tmp):", n_leave)
                return all_leaved, all_dropped

            # それ以外の場合は更新
            # 残すものは新しくする
            all_leaved = []
            for cluster, result, connected_indices in zip(
                    clusters, results, connected_indices_list):
                result = AnnealingSampler.update_vector_weight(
                    cluster, result, connected_indices)
                for i, vector in enumerate(cluster):
                    if bool(result[i]):
                        all_leaved.append(vector)
                    else:
                        all_dropped.append(vector)

            if iter == 0 and n_leave < minimum_limit:
                # 初回で下限を下回った場合は選別後を返す
                print("len(all_leaved_tmp):", n_leave)
                return all_leaved, all_dropped

        return all_leaved, all_dropped


def random_select(cluster: List[VectorWrapper], leave_rate: float):
    # ランダム選別を実施
    leave_size = math.ceil(len(cluster) * leave_rate)
    leaved_indices = random.sample(list(range(len(cluster))), leave_size)

    # 残すものと削除するものに分割
    all_leaved: List[VectorWrapper] = []
    all_dropped: List[VectorWrapper] = []
    all_leaved_data = []  # 検索用のndarray変数
    all_dropped_data = []  # 検索用のndarray変数
    for i, vector in enumerate(cluster):
        if i in leaved_indices:
            all_leaved.append(vector)
            all_leaved_data.append(vector.vector)
        else:
            all_dropped.append(vector)
            all_dropped_data.append(vector.vector)
    
    # 検索用に距離行列を作成
    all_leaved_data = np.array(all_leaved_data)
    all_dropped_data = np.array(all_dropped_data)
    dist_mat = pairwise_distances(all_leaved_data, all_dropped_data)
    nearest_ids = np.argmin(dist_mat, axis=0)

    # 削除するデータに最近傍の残すデータに重みを追加する
    for i, nearest_leaved_index in enumerate(nearest_ids):
        all_leaved[nearest_leaved_index].weight += all_dropped[i].weight
        all_dropped[i].parent_id = all_leaved[nearest_leaved_index].id
    
    return all_leaved, all_dropped


class RandomSelector:
    def __init__(self,
                 max_data_size: int = 100) -> None:
        self.max_data_size: int = max_data_size
        self.median_cut: MedianCut = MedianCut()

    def select(self, vectors: List[VectorWrapper], minimum_limit: int):
        leave_rate = minimum_limit / len(vectors)

        clusters: List[List[VectorWrapper]] = self.median_cut.divide(
            vectors, self.max_data_size)
        all_leaved = []
        all_dropped = []

        cluster: List[VectorWrapper]
        outputs = [random_select(cluster, leave_rate) for cluster in clusters]
        for result in outputs:
            all_leaved.extend(result[0])
            all_dropped.extend(result[1])
            
        return all_leaved, all_dropped


def kmeans_select(cluster: List[VectorWrapper], leave_rate: float):
    # 選別実行
    n_centers = math.ceil(len(cluster) * leave_rate)
    data = np.array([vector.vector for vector in cluster])
    kmeans = MiniBatchKMeans(n_clusters=n_centers, random_state=0, max_no_improvement=3).fit(data)
    labels = np.array(kmeans.labels_)  # 各データのクラスタID

    all_leaved: List[VectorWrapper] = []
    all_dropped: List[VectorWrapper] = []

    leaved_vector: VectorWrapper
    dropped_vector: VectorWrapper
    for label in range(n_centers):
        # クラスタに所属する点のもともとのID
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        # クラスタ中心に最近傍の点を取得
        cluster_center = kmeans.cluster_centers_[label]
        nearest_id_id = np.argmin(pairwise_distances(data[indices], [cluster_center]))
        nearest_id = indices[nearest_id_id]
        leaved_vector = cluster[nearest_id]
        # クラスタの他の点droppedにする
        dropped_ids = indices.tolist()
        dropped_ids.pop(nearest_id_id)  # 残すデータのIDを削除
        for dropped_id in dropped_ids:
            dropped_vector = cluster[dropped_id]
            leaved_vector.weight += dropped_vector.weight
            dropped_vector.parent_id = leaved_vector.id
            all_dropped.append(dropped_vector)

        all_leaved.append(leaved_vector)

    return all_leaved, all_dropped


class KMeansSelector:
    def __init__(self,
                 max_data_size: int = 100) -> None:
        self.max_data_size: int = max_data_size
        self.median_cut: MedianCut = MedianCut()
        pass

    def select(self, vectors: List[VectorWrapper], minimum_limit: int):
        leave_rate = minimum_limit / len(vectors)

        clusters: List[List[VectorWrapper]] = self.median_cut.divide(
            vectors, self.max_data_size)
        all_leaved = []
        all_dropped = []

        cluster: List[VectorWrapper]
        parallel_outputs = joblib.Parallel(n_jobs=-1)(joblib.delayed(kmeans_select)(
            cluster, leave_rate) for cluster in clusters)
        # parallel_outputs = [kmeans_select(cluster, leave_rate) for cluster in clusters]
        for result in parallel_outputs:
            all_leaved.extend(result[0])
            all_dropped.extend(result[1])
            
        return all_leaved, all_dropped
