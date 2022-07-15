import pandas as pd
import numpy as np

from libs.model import (
    get_data_set,
    CNNNet,
    train
)


def main():
    dataset, val_dataset = get_data_set()
    epochs = 20
    batch_size = 100
    n_output = 10  # クラス数
    log_for_each_iter = 50
    select_for_each_epoch = 5
    select_min_size = 10000

    net = CNNNet(n_output=n_output)
    net.cuda()
    history = train(net, epochs, dataset, val_dataset,
                    batch_size, log_for_each_iter=log_for_each_iter)
    columns = ["n_iter", "epoch", "train_loss", "val_loss"]
    df = pd.DataFrame(np.array(history), columns=columns)
    df.to_csv("hisotry_unselected.csv", index=False)

    net = CNNNet(n_output=n_output)
    net.cuda()
    history = train(net, epochs*5, dataset,
                    val_dataset, batch_size,
                    log_for_each_iter=log_for_each_iter,
                    will_select=True,
                    select_for_each_epoch=select_for_each_epoch,
                    select_min_size=select_min_size)
    df = pd.DataFrame(np.array(history), columns=columns)
    df.to_csv("hisotry_selected.csv", index=False)


if __name__ == "__main__":
    main()
