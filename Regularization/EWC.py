from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
import torch
from torch.autograd import Variable
import torch.utils.data
import random
import numpy as np

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(nn.Module):
    def __init__(self, model, opt, gamma, ewc_lambda, device='cuda', online=True, emp_FI=True):
        super(EWC, self).__init__()
        
        self.model = model
        self.ewc_lambda = ewc_lambda     #-> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = gamma         #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = online
        
        self.device = device
        self.emp_FI = emp_FI
        
        self.EWC_task_count = 0
        
        self.opt = opt
        
        self.model = self.model.to(device)
        
    def estimate_fisher(self, data_loader, allowed_classes=None, collate_fn=None):
        est_fisher_info = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()
        
        mode = self.model.training
        self.model.eval()        
        
        for index,(x,y) in enumerate(data_loader):
            x = x.to(self.device)
            output = self.model(x) if allowed_classes is None else self.model(x)[:, allowed_classes]
            
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y)==int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self.device)
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
                        
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}
        
        
        # Store new values in the network
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     p.detach().clone())
                
                
                if self.online and self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                    
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
                                     est_fisher_info[n])
        
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1
        
        self.model.train(mode=mode)
        
        
    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            for task in range(1, self.EWC_task_count+1):
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma*fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self.device)
        
    def forward(self, x):
        return self.model(x)
    

    def observe(self, data_loader, P=None):
        epoch_loss = 0
        self.model.train()
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
            penalty = self.ewc_lambda * self.ewc_loss()
            loss = F.cross_entropy(outputs, labels)
            assert not torch.isnan(loss)
            epoch_loss += loss.item()
            loss_total = loss+penalty
            loss_total.backward()
            self.opt.step()
    
        return epoch_loss / len(data_loader)  
