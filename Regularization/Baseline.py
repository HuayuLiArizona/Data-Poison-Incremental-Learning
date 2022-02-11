import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import torch
import torchvision

class Baseline(nn.Module):
    def __init__(self, model, opt, device="cuda:0"):
        super(Baseline, self).__init__()
        
        self.model = model
        self.opt = opt
        
        self.device = device
        
        self.model = self.model.to(device)
    
    def end_task(self, train_loader):
        print("doing nothing")
    
    def forward(self, x, task_id=None):
        return self.model(x, task_id)
                
    def observe(self, data_loader, P=None):
        self.model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if P is not None:
                xp, yp = next(iter(P))
                xp, yp = xp.to(self.device), yp.to(self.device)
                
                inputs = torch.cat([inputs, xp], 0)
                labels = torch.cat([labels, yp], 0)
                perm = torch.randperm(inputs.shape[0])
                
                inputs = inputs[perm]
                labels = labels[perm]
                
                
            self.opt.zero_grad()
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.opt.step()
    
        return epoch_loss / len(data_loader)  