import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import torch
import torchvision

class EWC(nn.Module):
    def __init__(self, backbone : nn.Module, loss : nn.Module, 
                 opt: torch.optim, importance: float, device: str):
        super(EWC, self).__init__()
        
        self.net = backbone
        self.loss = loss
        self.opt = opt
        
        self.importance = importance
        
        self.device = device
        
        self.net.to(self.device)
        
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoints = []
        self.fish = None        
        
    def penalty(self):
        if len(self.checkpoints)==0:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = 0
            for checkpoint in self.checkpoints:
                penalty += (self.fish * ((self.net.get_params() - checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, train_loader):
        fish = torch.zeros_like(self.net.get_params())
        
        for j, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.net.zero_grad()
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(train_loader) * train_loader.batch_size)

        self.fish = fish

        self.checkpoints.append(self.net.get_params().data.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def observe(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.importance * self.penalty()
        loss = self.loss(outputs, labels)
        assert not torch.isnan(loss)
        loss_total = loss+penalty
        loss_total.backward()
        self.opt.step()

        return loss.item()