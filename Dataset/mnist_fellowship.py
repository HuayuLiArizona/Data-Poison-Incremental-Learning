import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from torchvision.datasets import MNIST, QMNIST, KMNIST, FashionMNIST, EMNIST
from torchvision import datasets, transforms

from torch.utils.data import Subset, ConcatDataset, TensorDataset

import random

import torchvision.transforms.functional as TF

import numpy as np

train_data_list = [
    '/mnist_train_data.npy',
    '/fmnist_train_data.npy',
    '/kmnist_train_data.npy'
]

train_label_list = [
    '/mnist_train_label.npy',
    '/fmnist_train_label.npy',
    '/kmnist_train_label.npy'
]


val_data_list = [
    '/mnist_val_data.npy',
    '/fmnist_val_data.npy',
    '/kmnist_val_data.npy'
]

val_label_list = [
    '/mnist_val_label.npy',
    '/fmnist_val_label.npy',
    '/kmnist_val_label.npy'
]

def get_mnist_fellowship(root='./Dataset'):
    scenario_train = []
    scenario_test = []
    for dp,lp in zip(train_data_list, train_label_list):
        data = np.load(root+dp)
        data = torch.from_numpy(data).unsqueeze(1)
        label = np.load(root+lp)
        label = torch.from_numpy(label)
    
        scenario_train.append(TensorDataset(data, label))
    
    for dp,lp in zip(val_data_list, val_label_list):
        data = np.load(root+dp)
        data = torch.from_numpy(data).unsqueeze(1)
        label = np.load(root+lp)
        label = torch.from_numpy(label)
    
        scenario_test.append(TensorDataset(data, label))
    
    return scenario_train, scenario_test

def dummy():
    mnist_train = MNIST("./temp", download=True, train=True, transform = transforms.ToTensor())
    mnist_test = MNIST("./temp", download=True, train=False, transform = transforms.ToTensor())
    
    fmnist_train = FashionMNIST("./temp", download=True, train=True, transform = transforms.ToTensor())
    fmnist_test = FashionMNIST("./temp", download=True, train=False, transform = transforms.ToTensor())
    
    kmnist_train = KMNIST("./temp", download=True, train=True, transform = transforms.ToTensor())
    kmnist_test = KMNIST("./temp", download=True, train=False, transform = transforms.ToTensor())
    
    return [mnist_train, fmnist_train, kmnist_train], [mnist_test, fmnist_test, kmnist_test]