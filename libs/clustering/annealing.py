import openjij as oj
from typing import List
import numpy as np
import dwave.system
import yaml
try:
    import amplify
except:
    pass
from sklearn.metrics import pairwise_distances
import warnings
from scipy.sparse import SparseEfficiencyWarning

from .vector_wrapper import VectorWrapper

warnings.simplefilter('ignore', category=SparseEfficiencyWarning)


class CoarseQubo:
    def __init__(self, limit_coeff: float = 1.0):
        self.data_use_array = None
        self.model = None
        self.array_name = 'use'
        self.data_size = None
        self.limit_coefficient = limit_coeff
        self.connection_mat: np.ndarray = None
        self.connected_indices: List[List[int]] = None
        self.dist_mat: np.ndarray = None

    @staticmethod
    def is_near(vec1: VectorWrapper, vec2: VectorWrapper, near_limit: float):
        dist = vec1.dist(vec2)
        return dist < near_limit

    def build(self, near_limit: float, vectors: List[VectorWrapper]):

        self.data_size = len(vectors)
        qubo = {}
        all_connected_indices = {}

        # 距離行列を作成
        vectors_list = np.array([vec.vector for vec in vectors])
        self.dist_mat = pairwise_distances(vectors_list)
        self.connection_mat = self.dist_mat < near_limit

        connected_indices = np.where(self.connection_mat)
        for i, j in zip(*connected_indices):
            # QUBOのインデックスはpure pythonのintである必要がある。
            i = int(i)
            j = int(j)
            if i == j:
                qubo[(i, i)] = -vectors[i].weight
            else:
                qubo[(i, j)] = self.limit_coefficient
                if i not in all_connected_indices:
                    all_connected_indices[i] = []
                all_connected_indices[i].append(j)
        self.connected_indices = all_connected_indices
        self.qubo = qubo
        return self.qubo

    def decode_result(self, result):
        decoded_array = [result[i] for i in range(len(result))]
        return decoded_array


config_file = "tokens.yml"


class AnnealingSampler:
    def __init__(self, n_sample: int, limit_coeff: float = 100, use_dwave=False, use_amplify=False) -> None:
        self.use_dwave = use_dwave
        self.use_amplify = use_amplify
        if use_dwave:
            token_file = config_file
            with open(token_file) as f:
                token = yaml.safe_load(f)["dwave"]
            self.sampler = dwave.system.LeapHybridBQMSampler(token=token)
            self.use_amplify = False
        elif use_amplify:
            client = amplify.client.FixstarsClient()
            token_file = config_file
            with open(token_file) as f:
                client.token = yaml.safe_load(f)["fixstars"]
            self.sampler = amplify.Solver(client)
            self.amplify_model = None
            self.use_dwave = False
        else:
            self.sampler = oj.SASampler()
        self.model = None
        self.n_sample = n_sample
        self.qubo_creator: CoarseQubo = CoarseQubo(limit_coeff=limit_coeff)

    def setup_qubo(self, vectors: List[VectorWrapper], near_limit: float):
        self.model = self.qubo_creator.build(near_limit, vectors)
        if self.use_amplify:
            # QUBOの器を作成
            amplify_qubo = amplify.BinaryMatrix(len(vectors))
            for i, j in self.model:
                amplify_qubo[i, j] = self.model[(i, j)]
            self.amplify_model = amplify_qubo

    @property
    def connection_mat(self) -> np.ndarray:
        return self.qubo_creator.connection_mat

    @property
    def connected_indices(self) -> List[List[int]]:
        return self.qubo_creator.connected_indices

    @property
    def distance_mat(self) -> np.ndarray:
        return self.qubo_creator.dist_mat

    def solve(self):
        if self.use_amplify:
            solutions = self.sampler.solve(self.amplify_model)
            solution = solutions[0]  # 最適解のみ取得
            return solution.values

        # サンプリングを実行して最小パラメータを取得
        if self.use_dwave:
            response = self.sampler.sample_qubo(
                Q=self.model)
        else:
            response = self.sampler.sample_qubo(
                Q=self.model, num_reads=self.n_sample, sparse=True)
        response = response.first.sample
        # True, Falseの配列にして返す。
        result = self.qubo_creator.decode_result(response)
        return result

    @staticmethod
    def set_parent_vector(
            from_vector: VectorWrapper,
            all_vectors: List[VectorWrapper],
            will_leaved: List[bool],
            connected_indices: List[int]):
        # 最近傍の接続済みvectorを検索
        min_dist: float = None
        min_parent_id: int = None
        min_id: int = None
        assert len(connected_indices) > 0
        for i in connected_indices:
            if not will_leaved[i]:
                # 削除されるベクトルは排除
                continue

            connected_vector = all_vectors[i]
            dist = from_vector.dist(connected_vector)
            if min_dist is None:
                min_dist = dist
                min_parent_id = connected_vector.id
                min_id = i
            elif dist < min_dist:
                min_dist = dist
                min_parent_id = connected_vector.id
                min_id = i

        # 親ベクトルとしてidを登録
        assert min_dist is not None
        from_vector.parent_id = min_parent_id

        # 親ベクトルの方の重みを変更
        all_vectors[min_id].weight += from_vector.weight

    def select(self, vectors: List[VectorWrapper], near_limit: float):
        self.setup_qubo(vectors, near_limit)
        # 接続している要素がなければleavedのみになるはずなので処理をスキップ
        if not self.connected_indices:
            return [True] * len(vectors), self.connected_indices

        result = self.solve()
        return result, self.connected_indices

    @staticmethod
    def update_vector_weight(vectors: List[VectorWrapper], will_leaved: List[int], connected_indices: List[int]):
        # 事前にwill_leavedを整理
        for i, will_leave in enumerate(will_leaved):
            if will_leave:
                will_leaved[i] = True  # そのまま
            else:
                if i in connected_indices:
                    # connectedsのwill_leavedを調査
                    connecteds = connected_indices[i]
                    has_leaved_connected = False
                    for connected_index in connecteds:
                        if will_leaved[connected_index]:
                            has_leaved_connected = True
                            break
                    if not has_leaved_connected:
                        # 接続先もFalseなのであればミスとして、この点は残す
                        will_leaved[i] = True
                    else:
                        # それ以外は通常通り削除する
                        will_leaved[i] = False
                else:
                    # 接続する点がないのに削除されるのはミスなので戻す
                    will_leaved[i] = True
        
        for i, (vector, will_leave) in enumerate(zip(
                vectors, will_leaved)):
            if not will_leave:
                connecteds = connected_indices[i]

                # 削除するvectorに対して親vectorをセットする。
                AnnealingSampler.set_parent_vector(
                    vector, vectors, will_leaved, connecteds)
        return will_leaved
