from torch import nn
import torch    
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.body = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 10),
        )
        
    def forward(self, x):
        x = x.view(-1, 784)
        y = self.body(x)
        return y