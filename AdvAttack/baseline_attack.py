import numpy as np
import random
import torch
from torch.utils.data import Subset, ConcatDataset

def label_flip_attacks(dataset, percentage=10):
    
    sample_size = int(len(dataset)*(percentage*0.01))
    
    indices = list(range(len(dataset))) 
    random.shuffle(indices)
    indices = indices[:sample_size]
    
    for idx in indices:
        label = dataset.tensors[1][idx].detach()
        dataset.tensors[1][idx] = (label++torch.randint(1,10,(1,)))%10
    
    subset = Subset(dataset, indices)
    return subset