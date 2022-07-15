from libs.model import CNNNet

import torch


def main():
    n_output = 10  # クラス数

    net = CNNNet(n_output=n_output, will_add_dropout=False)
    print(net.will_add_dropout)
    net.load_state_dict(torch.load("random_model.pt"))
    print(net.will_add_dropout)
    # モデルに関わるものはメンバ変数として読み込まれてしまうが、プリミティブ型は問題なさそう。

if __name__ == "__main__":
    main()
