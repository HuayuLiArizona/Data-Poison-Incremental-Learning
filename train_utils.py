import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt
import random

from torch.utils.data import Subset, DataLoader


def get_one_class(dataset, target_class):
    indices = [index for index, element in enumerate(dataset.targets) if element == target_class]
    
    return Subset(dataset, indices)

def test_success_rate(model, dataset, origin_class, target_class):
    model.eval()
    correct = 0
    succes_rate = 0
    
    subset = get_one_class(dataset, origin_class)
    dataloader = DataLoader(subset, batch_size=1, num_workers=0)
    
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs, target = Variable(inputs).cuda(), Variable(target).cuda()
            output = model(inputs)
            succes_rate += (F.softmax(output, dim=1).max(dim=1)[1] == target_class).data.sum()
            correct += (F.softmax(output, dim=1).max(dim=1)[1] == origin_class).data.sum()
    return correct.cpu().numpy() / len(subset), succes_rate.cpu().numpy() / len(subset)


def mask_classes(outputs, task_id, class_per_task):
    outputs[:, 0:task_id * class_per_task] = -float('inf')
    outputs[:, (task_id + 1) * class_per_task:
               task_id * class_per_task] = -float('inf')

    return outputs


def test(model, data_loader, task_id, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, target in data_loader:
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)
            output = model(inputs)
            correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.cpu().numpy() / len(data_loader.dataset)

def test_class_incremental(model, data_loader, task_id, class_per_task, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, target in data_loader:
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)
            output = model(inputs)
            output = mask_classes(output,task_id, class_per_task)
            correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.cpu().numpy() / len(data_loader.dataset)

def loss_plot(x, epochs):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x, epochs, num_task):
    for t, v in x.items():
        plt.plot(list(range(int(t) * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)
    
    
def get_sample(dataset, sample_size):
        sample_idx = random.sample(range(len(dataset)), sample_size)
        return [img for img in dataset.train_data[sample_idx]]