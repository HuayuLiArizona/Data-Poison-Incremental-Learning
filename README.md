# Data-Poison-Incremental-Learning

This is a PyTorch implementation of the continual learning experiments on adversirial envioriments.

# Environment settings and libraries we used in our experiments
This project is tested under the following environment settings:
- '[google colab](https://colab.research.google.com/)'
- PyTorch: >= 1.6.0
- Torchvision: >= 0.6.0

# Running commands
### running on clean dataset
```python
!python -W ignore adv_si.py --dataset=clean --lr=0.0001 --batch_size=128 --epochs=10 --damping=0.01 --importance=5
!python -W ignore adv_ewc.py --dataset=clean --lr=0.0001 --batch_size=128 --epochs=10 --online=True --ewc_lambda=5000
!python -W ignore adv_ewc.py --dataset=clean --lr=0.0001 --batch_size=128 --epochs=10 --ewc_lambda=5000
 ````
### running on label flipping attacks
```python
!python -W ignore adv_si.py --dataset=lf --lr=0.0001 --batch_size=128 --epochs=10 --percentage=10 --damping=0.01 --importance=5
!python -W ignore adv_ewc.py --dataset=lf --lr=0.0001 --batch_size=128 --epochs=10 --percentage=10 --online=True --ewc_lambda=5000
!python -W ignore adv_ewc.py --dataset=lf --lr=0.0001 --batch_size=128 --epochs=10 --percentage=10 --ewc_lambda=5000
 ````
 
 ### running on our attacks
```python
!python -W ignore adv_si.py --dataset=adv --lr 0.0001 --batch_size 128 --epochs 10 --percentage 5 --damping 0.01 --importance 5 --num_steps 240 --decay 1.0 --epsilon 0.1 --rule 'adaptive'
!python -W ignore adv_ewc.py --dataset=adv --lr 0.0001 --batch_size 128 --epochs 10 --percentage 5 --online --ewc_lambda 5000 --num_steps 250 --decay 1.0 --epsilon 0.1 --rule 'adaptive'
!python -W ignore adv_ewc.py --dataset=adv --lr 0.0001 --batch_size 128 --epochs 10 --percentage 20 --ewc_lambda 5000 --num_steps 250 --decay 1.0 --epsilon 0.1  --rule 'fixed'
 ````
