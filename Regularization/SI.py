import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import torch
import torchvision
import random


class SI(nn.Module):
    def __init__(self, model, opt, epsilon, importance, device='cuda'):
        super(SI, self).__init__()
        
        self.model = model
        self.opt = opt
        
        self.epsilon = epsilon
        
        self.importance = importance
        
        self.device = device
        
        self.W = {}
        
        self.model = self.model.to(device)
        
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.W[n] = p.data.clone().zero_()
                self.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())                
        
    def surrogate_loss(self):
        try:
            losses = 0
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses += (omega *(p-prev_values)**2).sum()
            return losses
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self.device)

    def update_omega(self):
        #After completing training on a task, update the per-parameter regularization strength.
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.clone().detach().fill_(0)
                
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                
                omega_add = self.W[n]/(p_change**2 + self.epsilon)
                    
                omega_new = omega + omega_add
                
                self.W[n].zero_()
                
                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)
    
    def forward(self, x):
        return self.model(x)
    
    
    def observe(self, data_loader, P = None):
        epoch_loss = 0
        self.model.train()
        for i, (inputs, labels) in enumerate(data_loader):
            self.opt.zero_grad()
            self.model.zero_grad()
            old_params = {}
            
            # record old parameters
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    old_params[n] = p.clone().detach()
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if P is not None:
                xp, yp = next(iter(P))
                xp, yp = xp.to(self.device), yp.to(self.device)
                
                inputs = torch.cat([inputs, xp], 0)
                labels = torch.cat([labels, yp], 0)
                perm = torch.randperm(inputs.shape[0])
                
                inputs = inputs[perm]
                labels = labels[perm]
                
                
            outputs = self.model(inputs)
            
            loss = F.cross_entropy(outputs, labels)            
            assert not torch.isnan(loss)
            epoch_loss += loss.item()
            
            penalty = self.importance * self.surrogate_loss()
            loss_total = loss+penalty
            
            
            
            loss_total.backward()
            self.opt.step()
            
            #update W
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    delta = p.detach()-old_params[n]
                    self.W[n].add_(-p.grad*delta)
                
    
        return epoch_loss / len(data_loader)

    