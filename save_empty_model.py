from multiprocessing.spawn import import_main_path
from libs.model import (
    get_data_set,
    CNNNet,
    train
)

import torch


def main():
    dataset, val_dataset = get_data_set()
    batch_size = 200
    n_output = 10  # クラス数

    net = CNNNet(n_output=n_output)
    torch.save(net.state_dict(), "random_model.pt")

    net.cuda()
    pretrain_epochs = 3
    pretrain_max_iter = pretrain_epochs * len(dataset) // batch_size
    train(net, pretrain_max_iter, dataset, val_dataset,
          batch_size, log_for_each_iter=pretrain_max_iter,
          will_select=False, lr=1e-3)

    torch.save(net.state_dict(), "pretrained_model.pt")

if __name__ == "__main__":
    main()
