import torch
from torch.utils.data import Subset
from random import shuffle
import random
from torchvision.datasets import MNIST, QMNIST, KMNIST, FashionMNIST, EMNIST
from torchvision import datasets, transforms

def get_indice(targets, values):
    indices = [index for index, element in enumerate(targets) if element in values]
    
    return indices

def split_into_tasks(train_datasets, val_datasets, num_class, tasks):
    
    index_list = list(range(0,num_class))
    
    segs = num_class//tasks
    
    train_subsets = []
    val_subsets = []
    
    for ii in range(tasks):
        train_indices = get_indice(train_datasets.targets, index_list[ii*segs:(ii+1)*segs])
        val_indices = get_indice(val_datasets.targets, index_list[ii*segs:(ii+1)*segs])
        
        train_subsets.append(Subset(train_datasets, train_indices))
        val_subsets.append(Subset(val_datasets, val_indices))
        
    return train_subsets, val_subsets


def get_split_mnist(num_tasks):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    mnist_train = MNIST("./temp", download=True, train=True, transform = transform)
    mnist_test = MNIST("./temp", download=True, train=False, transform = transform)
    
    return split_into_tasks(mnist_train, mnist_test, 10, num_tasks)


def get_split_kmnist(num_tasks):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    mnist_train = KMNIST("./temp", download=True, train=True, transform = transform)
    mnist_test = KMNIST("./temp", download=True, train=False, transform = transform)
    
    return split_into_tasks(mnist_train, mnist_test, 10, num_tasks)


def get_split_fashion_mnist(num_tasks):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    mnist_train = KMNIST("./temp", download=True, train=True, transform = transform)
    mnist_test = KMNIST("./temp", download=True, train=False, transform = transform)
    
    return split_into_tasks(mnist_train, mnist_test, 10, num_tasks)