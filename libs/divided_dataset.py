import torchvision
from torch.utils.data import Dataset


class DividedDataset(Dataset):

    def __init__(self, dataset: Dataset, use_former: bool = True) -> None:
        self.all_dataset = dataset
        half_size: int = len(dataset) // 2
        self.indices: list
        if use_former:
            self.indices = list(range(0, half_size))
        else:
            self.indices = list(range(half_size, len(self.all_dataset)))

    def __getitem__(self, index: int):
        # dataからindexに対応するデータを取得しweightsを追加して返す
        use_index = self.indices[index]
        x, y = self.all_dataset[use_index]
        return x, y

    def __len__(self):
        return len(self.indices)
