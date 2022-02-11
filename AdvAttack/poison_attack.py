import torch
from torch.nn import functional as F
from torch import nn, autograd
from torch.autograd import Variable
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Subset, ConcatDataset, DataLoader, TensorDataset
from torch import optim
import sys
from torchvision import transforms

def craft_adv(net, X, label, X_target, label_target, epsilon, num_steps, decay=1.0, rule='adaptive', random_init=False):
    if rule == 'adaptive':
        step_size = 2.*epsilon/num_steps
    elif rule == 'fixed':
        step_size = 0.1*epsilon
    
    net.eval()
    
    X_pgd = Variable(X.data, requires_grad=True)
    momentum = torch.zeros_like(X_pgd).detach().cuda()
    if random_init:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    
    for idx in range(num_steps):
        with torch.enable_grad():
            loss_target = F.cross_entropy(net(X_target), label_target)
            target_grads = torch.autograd.grad(loss_target, net.parameters(), retain_graph=True, create_graph=True)
            net.zero_grad()
            
            loss_posion = F.cross_entropy(net(X_pgd), label)
            poison_grads = torch.autograd.grad(loss_posion, net.parameters(), retain_graph=True, create_graph=True)
            net.zero_grad()
            
            grads = 0
            for poison_grad, target_grad in zip(poison_grads, target_grads):
                grads -= F.cosine_similarity(target_grad.flatten(), poison_grad.flatten(), dim=0)
            
            X_grad = torch.autograd.grad(grads, X_pgd)[0]
            
            X_grad = X_grad / torch.mean(torch.abs(X_grad), dim=(1,2,3), keepdim=True)
            
            X_grad = X_grad + momentum*decay
            momentum = X_grad
            eta = step_size * X_grad.data.sign()
            
            X_pgd = Variable(X_pgd.data - eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            
    return X_pgd


def dataset_split(dataset, sample_size):
    index = list(range(len(dataset)))
    random.shuffle(index)
    rind = index[:sample_size]
    
    keep = list(set(index)-set(rind))
    
    dataset_k = Subset(dataset, keep)
    dataset_r = Subset(dataset, rind)
    
    return dataset_k, dataset_r

def attack_dataset(net, train_set, target_set, epsilon, num_steps, percentage, decay=1., rule='adaptive', device='cuda:0', random_init=True):
    sample_size = int(len(train_set)*(percentage*0.01))
    
    clean_set, poison_set = dataset_split(train_set, sample_size)
    
    poison_loader = DataLoader(poison_set, shuffle=True, num_workers=0, batch_size=500)
    target_loader = DataLoader(target_set, shuffle=True, num_workers=0, batch_size=500)
    
    poison_data = []
    poison_targets = []
    
    for _, (X, label) in tqdm(enumerate(poison_loader)):
        X_target, label_target = next(iter(target_loader))
        X_target = X_target.cuda()
        label_target = (label_target+torch.randint_like(label_target,1,10))%10
        label_target = label_target.cuda()
        
        X = X.cuda()
        label = label.cuda()
        
        X_adv = craft_adv(net, X, label, X_target, label_target, epsilon, num_steps, decay=decay, rule=rule, random_init=random_init)
        
        poison_data.append(X_adv.detach().cpu())
        poison_targets.append(label.detach().cpu())
    
    poison_data = torch.cat(poison_data, 0)
    
    poison_targets = torch.cat(poison_targets, 0) 
    
    P = TensorDataset(poison_data, poison_targets)
    
    return clean_set+P