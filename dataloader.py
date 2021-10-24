import torch
from torch.functional import split
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import numpy as np

def prep_dataloader(ds):
    mean = {
        'cifar100': (0.5071, 0.4865, 0.4409),
        'cifar10': (0.4914, 0.4822, 0.4465),
        'mnist': (0.1307,)
    }[ds]

    std = {
        'cifar100': (0.2673, 0.2564, 0.2762),
        'cifar10': (0.2023, 0.1994, 0.2010),
        'mnist': (0.3081,)
    }[ds]

    size = {
        'cifar100': 32,
        'cifar10': 32,
        'mnist': 28
    }[ds]

    in_channels = 1 if ds == 'mnist' else 3

    num_classes = {
        'cifar100': 100,
        'cifar10': 10,
        'mnist': 10
    }[ds]


    trans = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    transform_test = transforms.Compose(trans)
    transform_train = transforms.Compose(trans)

    train_dataset = getattr(datasets, ds.upper())(f'data/{ds}', train=True, download=True, transform=transform_train)
    test_dataset = getattr(datasets, ds.upper())(f'data/{ds}', train=False, download=True, transform=transform_test)

    batch_size = 18

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                                num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                            num_workers=0)

    return train_dataset, test_dataset, train_loader, test_loader, in_channels, size, num_classes


def toy_dataloader(args):
    def synthetic_data(w, b, m_examples, scale=1):
        X = torch.randn((m_examples, len(w)))
        X[:, 1] *= scale
        y = torch.mv(X, w)
        y += torch.normal(0, 0.5, y.shape)
        return X, y

    def initial(config, w=None, b=0.0, scale=1):
        m, n = config["m"], config['n']
        return synthetic_data(w, b, m, scale)

    config = {
        "m": 200,
        "n": 2,
        "batch_size": 200,
        "lr": 0.15,
        "n_epochs": 60,
        "optimizer": "SGD",
        "optim_hparas": {
            'lr': 0.015,         # for SGD and Adam
            'momentum': 0.9,
            'nesterov':True
        }
    }
    true_w = torch.tensor([1., 1.])
    X, y = initial(config, true_w, scale=2.5)
    train_dataset = torch.utils.data.TensorDataset(X, y)

    splits = []
    size = len(train_dataset) // args['K']
    for _ in range(args['K'] - 1):
        splits.append(size)
    splits.append(len(train_dataset) - size * (args['K'] - 1))

    train_datasets = torch.utils.data.random_split(train_dataset, splits)

    # print(f'Dataset: {len(train_dataset)}/{len(test_dataset)}')

    train_loaders = [torch.utils.data.DataLoader(dataset, 200, shuffle=True,
                                num_workers=0) for dataset in train_datasets]
    test_loader = torch.utils.data.DataLoader(train_dataset, 200, shuffle=False,
                            num_workers=0)

    return train_loaders, test_loader
