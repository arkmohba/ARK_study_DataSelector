from libs.model import (
    get_data_set,
    CNNNet,
    get_data_loader
)
from libs.selected_dataset import SelectedDataset


def main():
    dataset, _ = get_data_set()
    batch_size = 100
    log_for_each_iter = 300
    select_min_size = 10000

    n_output = 10  # クラス数
    net = CNNNet(n_output=n_output, grad_target="conv2")
    net.cuda()

    new_dataset = SelectedDataset(
        dataset, net, select_min_size, will_select=True,
        select_mode='annealing',
        pre_remove_size=30000)
    print("selected dataset size: ", len(new_dataset))

    # loader = get_data_loader(new_dataset, 1, shuffle=False)
    # for x, y, w in loader:
    #     print(y)
    #     break


if __name__ == '__main__':
    main()
