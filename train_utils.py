import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt



plt.style.use("seaborn-white")

def mask_classes(outputs: torch.Tensor, task_id, class_per_task) -> None:
    outputs[:, 0:task_id * class_per_task] = -float('inf')
    outputs[:, (task_id + 1) * class_per_task:
               task_id * class_per_task] = -float('inf')

    return outputs


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, target in data_loader:
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)
            output = model(inputs)
            correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.cpu().numpy() / len(data_loader.dataset)


def loss_plot(x, epochs):
    for t, v in x.items():
        plt.plot(list(range(t * epochs, (t + 1) * epochs)), v)

def accuracy_plot(x, epochs, num_task):
    for t, v in x.items():
        plt.plot(list(range(int(t) * epochs, num_task * epochs)), v)
    plt.ylim(0, 1)