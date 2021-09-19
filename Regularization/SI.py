import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import torch
import torchvision

class SI(nn.Module):
    def __init__(self, backbone : nn.Module, loss : nn.Module, 
                 lr : float, opt: torch.optim, importance : float, xi : float, device: str):
        super(SI, self).__init__()
        
        self.net = backbone
        self.loss = loss
        self.opt = opt
        self.lr = lr
        
        self.device = device
        
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        
        self.big_omega = None
        self.small_omega = 0
        
        self.importance = importance
        self.xi = xi
        
    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += self.small_omega.to(self.device) / ((self.net.get_params().data.to(self.device) - self.checkpoint) ** 2 + self.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def observe(self, inputs, labels):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.importance * penalty
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.opt.step()
        
        self.small_omega += self.lr * self.net.get_grads().data ** 2
        
        return loss.item()