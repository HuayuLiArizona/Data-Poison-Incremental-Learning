import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from torchvision.datasets import MNIST, QMNIST, KMNIST, FashionMNIST, EMNIST
from torchvision import datasets, transforms



def min_rot(task_number, num_task):
    max_rot = 90.
    min_rot = 0.
    return task_number / num_task * (max_rot - min_rot) + min_rot

def max_rot(task_number, num_task):
    max_rot = 90.
    min_rot = 0.
    return (task_number+1) / num_task * (max_rot - min_rot) + min_rot

def rot_degree(task_number, num_task):
    return (min_rot(task_number, num_task), max_rot(task_number, num_task))


def get_rotate_mnist(num_tasks):
    scenario_train = []
    scenario_test = []
    for task_id in range(num_tasks):
        transform = transforms.Compose([transforms.Resize((32, 32)),transforms.RandomRotation(rot_degree(task_id, num_tasks)), transforms.ToTensor()])
        mnist_train = MNIST("./temp", download=True, train=True, transform = transform)
        scenario_train.append(mnist_train)
        scenario_test.append(MNIST("./temp", download=True, train=False, transform = transform))

    return scenario_train, scenario_test
