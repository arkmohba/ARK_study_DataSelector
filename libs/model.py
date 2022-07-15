import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import tqdm
import time
import joblib
import copy
import dataclasses

from .selected_dataset import GradsGettableNet, SelectedDataset


@dataclasses.dataclass
class TrainMetric:
    n_iter: int
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    min_val_loss: float
    max_val_acc: float
    select_time: float
    epoch_time: float
    train_data_len: int
    min_weights: float
    mean_weights: float
    max_weights: float
    effective_lr: float


def get_data_set() -> Dataset:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)),
         transforms.RandomAffine([-15, 15], scale=(0.8, 1.2), shear=10)
         ])

    train_set: Dataset = torchvision.datasets.MNIST(root='./data',
                                                    train=True,
                                                    download=True,
                                                    transform=transform)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    validation_set: Dataset = torchvision.datasets.MNIST(root='./data',
                                                         train=False,
                                                         download=True,
                                                         transform=transform)
    return train_set, validation_set


def get_data_loader(dataset, batch_size: int, shuffle=True):
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=6)
    return train_loader


def get_grad(duplicated_nets, inputs, labels, net_id, as_scalar: bool = False):
    net = duplicated_nets[net_id]
    # net.cuda()
    inputs = inputs.cuda()
    labels = labels.cuda()
    net.zero_grad()
    outputs = net(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    weight_grad = torch.flatten(net.grad_target.weight.grad)
    bias_grad = torch.flatten(net.grad_target.bias.grad)
    grad = torch.cat([weight_grad, bias_grad])
    if as_scalar:
        grad = torch.linalg.norm(grad)
    grad = grad.cpu().detach().numpy().copy()
    return grad


class CNNNet(GradsGettableNet):
    gattable_layers_name = ["conv1", "conv2", "fc2"]

    def __init__(self, n_output, grad_target: str = None, will_add_dropout: bool = True):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3)  # 26x26x64 -> 24x24x64
        self.pool = nn.MaxPool2d(2, 2)  # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, n_output)
        self.gettable_layers = {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "fc1": self.fc1,
            "fc2": self.fc2
        }
        self.grad_target = None
        if grad_target:
            self.grad_target = self.gettable_layers[grad_target]
        self.will_add_dropout = will_add_dropout

    def set_grad_target(self, grad_target: str):
        self.grad_target = self.gettable_layers[grad_target]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        if self.will_add_dropout:
            x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        if self.will_add_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def get_grads(self, dataset: Dataset, as_scalar: bool = False):
        """バッチサイズ1でシャッフル無効にしてデータローダーを作成し、gradを取得する"""
        data_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=2)
        self.cpu()
        n_jobs = 2
        duplicated_nets = [copy.deepcopy(self).cuda() for _ in range(n_jobs)]
        grads = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(get_grad)(
                duplicated_nets, inputs, labels, i % n_jobs, as_scalar)
            for i, (inputs, labels) in enumerate(tqdm.tqdm(data_loader)))
        for net in duplicated_nets:
            net.cpu()
        del duplicated_nets
        self.cuda()
        grads = np.nan_to_num(np.array(grads), nan=0, posinf=0, neginf=0)
        return grads


def validate(net, data_loader: DataLoader):
    net.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction='none')
        sum_loss = 0
        sum_count = 0
        sum_accuracy = 0
        for inputs, labels, _ in data_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward
            outputs = net(inputs)
            # loss
            loss = criterion(outputs, labels)
            loss = torch.sum(loss)
            sum_loss += loss.item()
            # accuracy
            y_pred_label = torch.max(outputs, 1)[1]
            sum_accuracy += torch.sum(y_pred_label == labels).item()
            # count
            sum_count += len(labels)

    net.train()
    return sum_loss / sum_count, sum_accuracy / sum_count


def create_optimizer(net, opt: str, lr: float):
    optimizers = {}
    optimizers['adadelta'] = optim.Adadelta
    optimizers['adagrad'] = optim.Adagrad
    optimizers['adam'] = optim.Adam
    optimizers['adamw'] = optim.AdamW
    optimizers['adamax'] = optim.Adamax
    optimizers['asgd'] = optim.ASGD
    optimizers['lbfgs'] = optim.LBFGS
    optimizers['nadam'] = optim.NAdam
    optimizers['radam'] = optim.RAdam
    optimizers['rmsprop'] = optim.RMSprop
    optimizers['rprop'] = optim.Rprop
    optimizers['sgd'] = optim.SGD
    # all_opts = ['adadelta', 'adagrad', 'adam', 'adamw', 'adamax', 'asgd', 'lbfgs', 'nadam', 'radam', 'rmsprop', 'rprop', 'sgd']
    assert opt in optimizers
    optimizer = optimizers[opt](net.parameters(), lr=lr)
    return optimizer


def train(net, max_iter, origin_train_dataset, val_dataset,
          batch_size: int,
          log_for_each_iter: int,
          will_select: bool = False,
          select_for_each_epoch: int = 1,
          select_min_size: int = None,
          select_mode: str = 'annealing',
          lr: float = 1e-3,
          opt: str = "adam",
          do_validate: bool = True):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = create_optimizer(net, opt, lr)

    train_dataset = SelectedDataset(origin_train_dataset, will_select=False)
    train_loader = get_data_loader(train_dataset, batch_size)

    validate_batch_size = 1500
    val_train_loader = get_data_loader(
        train_dataset, validate_batch_size, shuffle=False)
    val_dataset = SelectedDataset(val_dataset, will_select=False)
    val_loader = get_data_loader(
        val_dataset, validate_batch_size, shuffle=False)

    select_time = 0
    epoch_time = 0
    history = []
    n_iter = 0
    epoch = 0
    min_loss = None
    max_acc = None
    while True:
        # 特定のepochでデータセット自体を更新し、データローダを更新する
        if will_select and epoch % select_for_each_epoch == 0:
            start = time.time()
            del train_dataset
            del train_loader
            train_dataset = SelectedDataset(
                origin_train_dataset, net, select_min_size, will_select=will_select, select_mode=select_mode)
            train_loader = get_data_loader(train_dataset, batch_size)
            select_time = time.time() - start
        start = time.time()
        tmp_history, n_iter, min_loss, max_acc = train_for_1epoch(
            net, criterion, optimizer,
            train_loader, val_train_loader, val_loader,
            len(train_dataset),
            max_iter, epoch, n_iter,
            log_for_each=log_for_each_iter,
            select_time=select_time,
            epoch_time=epoch_time,
            lr=lr,
            min_loss=min_loss,
            max_acc=max_acc,
            do_validate=do_validate)
        epoch_time = time.time() - start
        history.extend(tmp_history)
        if n_iter >= max_iter:
            break
        epoch += 1

    return history


def train_for_1epoch(
        net, criterion, optimizer,
        train_loader: DataLoader,
        val_train_loader: DataLoader,
        val_loader: DataLoader,
        train_data_len: int,
        max_iter: int,
        epoch: int,
        n_initial_iter: int,
        log_for_each: int,
        select_time: float,
        epoch_time: float,
        lr: float,
        min_loss: float,
        max_acc: float,
        do_validate: bool = True):
    n_iter = n_initial_iter
    history = []
    weights = []
    for inputs, labels, weight in tqdm.tqdm(train_loader):
        weights.extend(weight.tolist())
        inputs = inputs.cuda()
        labels = labels.cuda()
        weight = weight.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) * weight
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()

        if do_validate and n_iter % log_for_each == 0:
            train_loss = 0
            train_acc = 0
            # train_loss, train_acc = validate(net, val_train_loader)
            val_loss, val_acc = validate(net, val_loader)
            min_loss = val_loss if min_loss is None else min(
                min_loss, val_loss)
            max_acc = val_acc if max_acc is None else max(max_acc, val_acc)
            weights = np.array(weights)
            weight_min = weights.min()
            weight_mean = weights.mean()
            weight_max = weights.max()
            effective_lr = lr * weight_mean
            metric = TrainMetric(n_iter, epoch, train_loss, train_acc,
                                 val_loss, val_acc, min_loss, max_acc, select_time,
                                 epoch_time, train_data_len,
                                 weight_min, weight_mean, weight_max,
                                 effective_lr)
            weights = []
            history.append(metric)
        n_iter += 1
        if n_iter >= max_iter:
            break
    return history, n_iter, min_loss, max_acc
