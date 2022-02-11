import Dataset.mnist_fellowship as mnist_dataset
from Regularization.Baseline import Baseline
from Backbones.mnist_model import CNN,MLP
import random
from torch import optim
from tqdm import tqdm

from train_utils import test, test_success_rate

from torch.utils.data import DataLoader, Subset
from AdvAttack.poison_attack import attack_dataset
from AdvAttack.baseline_attack import label_flip_attacks
import json

from weight_initializer import Initializer

import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser(description='Run baseline experiment.')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
parser.add_argument('--epochs', type=int, default=20, help='epochs for training')
parser.add_argument('--num_task', type=int, default=3, help='numbers of tasks to run')
parser.add_argument('--dataset', type=str, default='clean', help='choose dataset: 1.clean 2.lf 3.adv')
parser.add_argument('--percentage', type=int, default=10, help='percentage of data to poison')
parser.add_argument('--num_steps', type=int, default=40, help='PGD num steps')
parser.add_argument('--epsilon', type=float, default=16/255, help='PGD epsilon')
parser.add_argument('--decay', type=float, default=1.0, help='mPGD decay')
def train_baseline(args):
    backbone = MLP()
    
    Initializer.initialize(model=backbone)
    
    optimizer = optim.Adam(params=backbone.parameters(), lr=args.lr)
    
    baseline = Baseline(model=backbone, opt=optimizer)
    
    scenario_train, scenario_test = mnist_dataset.get_mnist_fellowship()
    
    test_loader = {}
    for i in range(args.num_task):
        test_loader[i] = DataLoader(scenario_test[i], batch_size=args.batch_size, num_workers=0)
        
    acc = {}
    
    
    
    acc = {}
    for task in range(args.num_task):
        acc[task] = []
        
        if task == 0:
            dataset_train = scenario_train[task]   
            train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
            P = None
        else:
            if args.dataset == 'adv':
                dataset_train = attack_dataset(baseline.model, scenario_train[task], scenario_train[0],
                               args.epsilon, args.num_steps, args.percentage, args.decay)
                train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
                P = None
            elif args.dataset == 'lf':
                dataset_poison = label_flip_attacks(scenario_train[0], percentage=args.percentage)
                dataset_train = scenario_train[task] +dataset_poison
                train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
                P = None
            else:
                dataset_train = scenario_train[task]   
                train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
                P = None
        
        for _ in tqdm(range(args.epochs)):
            _ = baseline.observe(train_loader, P)
            
            for sub_task in range(task + 1):
                acc[sub_task].append(test(baseline.model, test_loader[sub_task], task_id=sub_task, device='cuda'))
    
    return acc
    
def get_time():
    x = str(datetime.now())
    x = x.replace(':', '-').replace(' ', '-').replace('.', '-')
    
    return x


if __name__ == "__main__":
    
    args = parser.parse_args()
    print(args)
    
    roots = './Logs/baseline/'
    if os.path.isdir(roots):
        pass
    else:
        os.mkdir(roots)
        
    roots += args.dataset
    
    if args.dataset != 'clean':
        roots += str(args.percentage)
    
    if os.path.isdir(roots):
        pass
    else:
        os.mkdir(roots)
    
    log_path = roots+'/'+get_time()+'.json'

    acc = train_baseline(args)
    
    with open(log_path, 'w') as fp:
        json.dump(acc, fp)