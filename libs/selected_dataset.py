
from typing import List
from torch.utils.data import Dataset, Subset
import torch.nn as nn
import numpy as np

from .clustering.data_optimizer import DataOptimizer, RandomSelector, KMeansSelector
from .clustering.vector_wrapper import VectorWrapper


class GradsGettableNet(nn.Module):
    """

    Args:
        nn (_type_): _description_
    """

    def get_grads(self, dataset: Dataset, as_scalar: bool = False, indices: List[int] = None):
        raise NotImplementedError


class SelectedDataset(Dataset):

    def __init__(self, dataset: Dataset, net: GradsGettableNet = None, select_min_size: int = None, will_select: bool = True, select_mode: str = 'annealing', pre_remove_size: int = None) -> None:
        self.all_dataset = dataset
        self.will_select = will_select
        if self.will_select:
            assert net is not None
            assert select_min_size is not None
            indices_sorted = None
            if pre_remove_size:
                # データの微分の絶対値を計算
                grad_scales: List[float] = net.get_grads(dataset, as_scalar=True)
                grad_scales = np.array(grad_scales)
                # 微分値でargsort
                indices_sorted = np.argsort(grad_scales)
                # 微分値が小さいインデックスを削除
                indices_sorted = indices_sorted[pre_remove_size:]
                # インデックス自体の数値でソート
                indices_sorted = np.sort(indices_sorted)
                # サブセット化
                self.all_dataset = Subset(self.all_dataset, indices=indices_sorted)
                print("reduced size:",len(self.all_dataset))

            # 残ったインデックスの微分ベクトルを再度取得
            # データに対して微分ベクトルを計算
            vectors = net.get_grads(self.all_dataset)
            vectors = VectorWrapper.create_vector_list(vectors)

            # 微分をもとにデータを選別し、そのインデックスを保持する
            if select_mode == 'random':
                max_data_size = 500
                data_optimizer = RandomSelector(max_data_size)
            elif select_mode == 'kmeans':
                max_data_size = 500
                data_optimizer = KMeansSelector(max_data_size)
            else:
                annealing_max_data_size = 200
                data_optimizer = DataOptimizer(
                    max_data_size=annealing_max_data_size, n_sample=10)

            print("select min size:", select_min_size)
            leaved: List[VectorWrapper]
            leaved, _ = data_optimizer.select(
                vectors, minimum_limit=select_min_size)
            print("leaved size:",len(leaved))
            self.indices = [vector.id for vector in leaved]
            self.all_dataset = Subset(self.all_dataset, self.indices)
            self.weights: List[float] = [
                vector.weight for vector in leaved]
            self.weights: np.ndarray = np.array(
                self.weights).astype(np.float64)
            # self.weights = self.weights / self.weights.mean()
        else:
            self.indices = list(range(len(self.all_dataset)))
            self.weights = [1] * len(self.all_dataset)

    def __getitem__(self, index: int):
        # dataからindexに対応するデータを取得しweightsを追加して返す
        # use_index = self.indices[index]
        # x, y = self.all_dataset[use_index]
        x, y = self.all_dataset[index]
        weight = self.weights[index]
        return x, y, weight

    def __len__(self):
        return len(self.all_dataset)
