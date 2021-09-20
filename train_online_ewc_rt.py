from Dataset.Rotate_MNIST import get_rotate_mnist
from Regularization.OnlineEWC import OnlineEWC
from Backbones.LeNet import LeNet

import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from train_utils import test, loss_plot, accuracy_plot

from torch.utils.data import DataLoader

import json

def train_online_ewc_rotate_mnist(lr = 0.01, batch_size = 256, epochs=20, num_task=5):
    backbone = LeNet()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=backbone.parameters(), lr=lr)
    
    ewc_on = OnlineEWC(backbone=backbone, loss=loss, opt=optimizer, gamma=0.95, importance=0.5, device='cuda:0')
    
    
    scenario_train, scenario_test = get_rotate_mnist(num_tasks=5)
    
    train_loader = {}
    test_loader = {}
    for i in range(num_task):
        train_loader[i] = DataLoader(scenario_train[i], batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader[i] = DataLoader(scenario_test[i], batch_size=batch_size, num_workers=0)
    
    acc = {}
    for task in range(num_task):
        acc[task] = []
        
        for _ in tqdm(range(epochs)):
            for i, (inputs, labels) in enumerate(train_loader[task]):
                loss = ewc_on.observe(inputs, labels)
                
            for sub_task in range(task + 1):
                acc[sub_task].append(test(ewc_on.net, test_loader[sub_task], device='cuda:0'))
            
        ewc_on.end_task(train_loader[task])
   
    return acc
    
   
if __name__ == "__main__":
    acc = train_online_ewc_rotate_mnist(lr = 0.01, batch_size = 256, epochs=20, num_task=5)

    with open('./logs/online_ewc_rotate_mnist.json', 'w') as fp:
        json.dump(acc, fp)
