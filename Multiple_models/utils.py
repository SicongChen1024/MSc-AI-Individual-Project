#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 21:17:07 2021

@author: chrischen
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import random

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def check_accuracy(loader, model, attack, adv_test=False):
    
 
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            if adv_test:
              with torch.enable_grad():
                adv_x = attack.perturb(x, y, 'mean', False)
                x = adv_x.detach()

            scores = model(x)
              
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc

def check_accuracy_torchattack(loader, model, attack):
    
 
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            
            with torch.enable_grad():
              adv_x = attack(x, y)
              x = adv_x.detach()

            scores = model(x)
              
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc

def check_acc_rand(loader, model_lst, attack, adv_test=False):

  num_correct = 0
  num_samples = 0
  n = len(model_lst)
  for i in range(n):
    model_lst[i].eval()

  with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device, dtype=dtype)  # move to device
          y = y.to(device=device, dtype=torch.long)

          index = random.randint(0,n-1)
          #print('index: %d'%(index))
          model_i = model_lst[index]

          if adv_test:
            with torch.enable_grad():
              adv_x = attack.perturb(x, y, 'mean', False)
              x = adv_x.detach()

          #scores = model(x)
          scores = model_i(x)
            
          _, preds = scores.max(1)
          num_correct += (preds == y).sum()
          num_samples += preds.size(0)
      acc = float(num_correct) / num_samples
      print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

  return acc

def check_acc_rand_and_combined(loader, model_adv_lst, model_JR_lst, attack, JR_ratio=1/2,  adv_test=False):

  num_correct = 0
  num_samples = 0
  n = len(model_adv_lst)
  for i in range(n):
    model_adv_lst[i].eval()
    model_JR_lst[i].eval()

  with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device, dtype=dtype)  # move to device
          y = y.to(device=device, dtype=torch.long)

          adv_index = random.randint(0,n-1)
          JR_index = random.randint(0,n-1)
          model_adv_i = model_adv_lst[adv_index]
          model_JR_i = model_JR_lst[JR_index]

          if adv_test:
            with torch.enable_grad():
              adv_x = attack.perturb(x, y, 'mean', False)
              x = adv_x.detach()

          #scores = model(x)
          #scores = (model_adv_i(x) + model_JR_i(x))/2
          scores = JR_ratio*model_JR_i(x) + (1-JR_ratio)*model_adv_i(x)
            
          _, preds = scores.max(1)
          num_correct += (preds == y).sum()
          num_samples += preds.size(0)
      acc = float(num_correct) / num_samples
      print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

  return acc

  
