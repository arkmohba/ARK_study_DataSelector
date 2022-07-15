import traceback
from typing import final

from libs.model import (
    get_data_set,
    CNNNet,
    train,
    TrainMetric
)
from libs.divided_dataset import DividedDataset
import torch

import mlflow
from sklearn.model_selection import ParameterGrid

mlflow.set_experiment('mnist_experiment_19_former_latter')


def main():
    origin_dataset, val_dataset = get_data_set()

    epochs = 20
    n_output = 10  # クラス数
    log_for_each_iter = 25
    will_select = True

    params_config = {
        'lr': [1e-5],
        'will_add_dropout': [True],
        'pretrain': ["former"],
        'dataset_type': ["latter", "all"],
        'batch_size': [200],
        'optimizer': ['sgd'],
        'layer_name': ['conv1', 'fc2'],
        'select_for_each_epoch': [7],
        'select_min_size': [10000],
        'select_mode': ['random', 'kmeans', 'annealing']
    }

    for i in range(3):
        for params in ParameterGrid(params_config):
            lr = params['lr']
            will_add_dropout = params['will_add_dropout']
            pretrain = params['pretrain']
            batch_size = params['batch_size']
            optimizer = params["optimizer"]
            layer_name = params["layer_name"]
            select_min_size = params["select_min_size"]
            select_for_each_epoch = params["select_for_each_epoch"]
            select_mode = params['select_mode']
            dataset_type = params['dataset_type']

            if dataset_type == 'former':
                dataset = DividedDataset(origin_dataset, use_former=True)
            elif dataset_type == 'latter':
                dataset = DividedDataset(origin_dataset, use_former=False)
            else:
                dataset = origin_dataset

            run_name = ":".join(
                ["Selected", select_mode, pretrain, dataset_type])
            run_name = run_name + "_" + str(i)
            max_iter = epochs * len(dataset) // batch_size
            print(f"START:{run_name}")
            try:
                with mlflow.start_run(run_name=run_name):
                    net = CNNNet(n_output=n_output,
                                 will_add_dropout=will_add_dropout)
                    load_model = ""
                    if pretrain == "former":
                        load_model = "former_pretrained_model.pt"
                    elif pretrain == "all":
                        load_model = "pretrained_model.pt"
                    else:
                        load_model = "random_model.pt"

                    net.load_state_dict(torch.load(load_model))
                    net.set_grad_target(layer_name)
                    net.cuda()
                    mlflow.log_param("grad_layer", layer_name)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("will_select", will_select)
                    mlflow.log_param("select_for_each_epoch",
                                     select_for_each_epoch)
                    mlflow.log_param("select_min_size", select_min_size)
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("will_add_dropout", will_add_dropout)
                    mlflow.log_param("pretrain", pretrain)
                    mlflow.log_param("optimizer", optimizer)
                    mlflow.log_param("select_mode", select_mode)
                    mlflow.log_param("train_dataset_type", dataset_type)
                    history = train(net, max_iter, dataset,
                                    val_dataset, batch_size,
                                    log_for_each_iter=log_for_each_iter * 200 // batch_size,
                                    will_select=True,
                                    select_for_each_epoch=select_for_each_epoch,
                                    select_min_size=select_min_size,
                                    select_mode=select_mode,
                                    lr=lr,
                                    opt=optimizer)
                    metric: TrainMetric
                    for metric in history:
                        mlflow.log_metric(key="train_loss",
                                          value=metric.train_loss,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="train_acc",
                                          value=metric.train_acc,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="val_loss",
                                          value=metric.val_loss,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="val_acc",
                                          value=metric.val_acc,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="epoch",
                                          value=metric.epoch,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="n_iter",
                                          value=metric.n_iter,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="select_time",
                                          value=metric.select_time,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="epoch_time",
                                          value=metric.epoch_time,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="dataset_size",
                                          value=metric.train_data_len,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="min_weight",
                                          value=metric.min_weights,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="mean_weight",
                                          value=metric.mean_weights,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="max_weight",
                                          value=metric.max_weights,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="effective_lr",
                                          value=metric.effective_lr,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="min_val_loss",
                                          value=metric.min_val_loss,
                                          step=metric.n_iter)
                        mlflow.log_metric(key="max_val_acc",
                                          value=metric.max_val_acc,
                                          step=metric.n_iter)
                    del net  # 念の為削除
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                if net:
                    del net
                continue
                # exit(1)
            finally:
                print(f"END:{run_name}")


if __name__ == "__main__":
    main()
