from libs.model import (
    get_data_set,
    CNNNet,
    train
)
from libs.divided_dataset import DividedDataset

import torch


def main():
    dataset, val_dataset = get_data_set()
    batch_size = 200
    n_output = 10  # クラス数

    net = CNNNet(n_output=n_output)

    net.cuda()
    pretrain_epochs = 20
    pretrain_max_iter = pretrain_epochs * len(dataset) // batch_size
    former_dataset = DividedDataset(dataset, use_former=True)
    train(net, pretrain_max_iter, former_dataset, val_dataset,
          batch_size, log_for_each_iter=pretrain_max_iter,
          will_select=False, lr=1e-3, do_validate=False)

    torch.save(net.state_dict(), "former_pretrained_model.pt")

if __name__ == "__main__":
    main()
