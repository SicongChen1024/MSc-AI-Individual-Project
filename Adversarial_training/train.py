#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
part of this code is modified from https://github.com/louis2889184/pytorch-adversarial-training
author: YI-LIN SUNG
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
from val import val_evaluation
sys.path.append("..") 
from Attack.PGD import PGD

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_every = 100
def train(model, train_loader, val_loader, optimizer, scheduler, path, epsilon=4/255, alpha=2/255, k=10, epochs=10, adv_train=False):
  if adv_train:
    print('Adversarial training')
    #standardval_loss_lst = []

  attack = PGD(model, 
            epsilon, 
            alpha, 
            min_val=-1, 
            max_val=1, 
            max_iters=k, 
            _type='linf')

  train_loss_lst = []
  val_loss_lst = []
  model = model.to(device=device)
  best_acc = .0
  num_samples = 0
  for e in range(epochs):
        total_train_loss = 0
        for t, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            num_samples += y.size(0)

            if adv_train:
              adv_x = attack.perturb(x, y, 'mean', True)
              scores = model(adv_x)
            else:
              scores = model(x)
              

            

            loss = F.cross_entropy(scores, y, reduction='sum')

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            total_train_loss += loss.item()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                #print()

        scheduler.step()

        train_loss_lst.append(total_train_loss/num_samples)
        if adv_train:
          current_acc, avg_val_loss = val_evaluation(val_loader, model, attack, adv_test=True)
          #standard_acc, standardavg_val_loss = val_evaluation(val_loader, model, adv_test=False)
          #standardval_loss_lst.append(standardavg_val_loss)
        else:
          current_acc, avg_val_loss = val_evaluation(val_loader, model, attack, adv_test=False)
        
        if current_acc > best_acc:
          best_acc = current_acc
          if adv_train:
            torch.save(model.state_dict(),  path + 'adv_' + str(int(epsilon*255)) + '_SqueezeNet.pt')
          else:
            torch.save(model.state_dict(), path + 'standard_SqueezeNet.pt')
        val_loss_lst.append(avg_val_loss)

  if not os.path.exists('../results'):
        os.makedirs('../results')
  epoch_lst = list(range(epochs))
  plt.plot(epoch_lst, train_loss_lst, label='train')
  plt.plot(epoch_lst, val_loss_lst, label='val')
  plt.xlabel('epoch')
  plt.ylabel('avg loss')
  plt.title('Loss curve')
  plt.legend()
  plt.savefig('../results/adv_train_val.png')
    
  return train_loss_lst, val_loss_lst
