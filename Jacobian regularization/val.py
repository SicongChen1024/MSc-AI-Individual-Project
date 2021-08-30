#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:23:04 2021

@author: chrischen
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from jacobian_reg import JacobianReg

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def val_evaluation(loader, model, attack, adv_test=False, jacobian_reg = False, lambda_JR = .1):
    
    reg_full = JacobianReg(n=-1)
    num_correct = 0
    num_samples = 0
    num_positive = 0
    model.eval()  # set model to evaluation mode
    loss_avg = 0
    if jacobian_reg:
      loss_super_avg = 0 
      loss_JR_avg = 0
    #with torch.no_grad():
    for x, y in loader:
        x = x.to(device=device, dtype=dtype)  # move to device
        y = y.to(device=device, dtype=torch.long)
        
        #print('Original images')
        #print(x)
        
        if adv_test:
          with torch.enable_grad():
            adv_x = attack.perturb(x, y, 'mean', False)
            x = adv_x.detach()
            
        x.requires_grad = True
        scores = model(x)
        loss_super = F.cross_entropy(scores, y)

        if jacobian_reg:
            
          loss_JR = reg_full(x, scores)
          loss = loss_super + lambda_JR * loss_JR
          loss_super_avg += loss_super.item() * y.size(0)
          loss_JR_avg += loss_JR.item() * y.size(0)
        else:
          loss = loss_super
        
        
        loss_avg += loss.item() * y.size(0)

        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_positive += (preds == 1).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Positive')
    print(num_positive)
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    
    loss_avg /= num_samples

    if jacobian_reg:
      loss_super_avg /= num_samples
      loss_JR_avg /= num_samples

      return acc, loss_super_avg, loss_JR_avg, loss_avg
    
    else:
      return acc, loss_avg