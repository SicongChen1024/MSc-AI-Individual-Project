#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
part of this code is modified from https://github.com/facebookresearch/jacobian_regularizer
author: Facebook Research
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from val import val_evaluation
from jacobian_reg import JacobianReg

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
print_every = 100
reg = JacobianReg(n=-1)
def train(model, model_name, train_loader, val_loader, attack, optimizer, scheduler, path, epochs=10, jacobian_reg = False, lambda_JR = .1):
  
  
  train_loss_lst = []
  val_loss_lst = []
  

  if jacobian_reg:
    print('Jacobian Regularization')
    super_loss_lst = []
    JR_loss_lst = []
    #loss_lst = []
 
  model = model.to(device=device)
  best_acc = .0
  num_samples = 0
  for e in range(epochs):
        train_loss_avg = 0
        for t, (x, y) in enumerate(train_loader):
            x = Variable(x)
            y = Variable(y)
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            num_samples += y.size(0)
      
              
            x.requires_grad = True
            scores = model(x)
              

            

            loss_super = F.cross_entropy(scores, y)

            if jacobian_reg:
              loss_JR = reg(x, scores)
              loss = loss_super + lambda_JR * loss_JR
            else:
              loss = loss_super
            

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            train_loss_avg += loss.item() * y.size(0)

            if t % print_every == 0:
                if jacobian_reg:
                  print('Epoch: %d, Iteration %d, supervised loss = %.4f, Jacobian loss = %.4f, total loss = %.4f' % (e, t, loss_super.item(), loss_JR.item(), loss.item()))
                else:
                  print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                #print()

        scheduler.step()

        #evaluation every epoch
        #print('Model parameters')
        #print(model.parameters())
        
        
        if jacobian_reg:
          current_acc, loss_super_avg, loss_JR_avg, loss_avg = val_evaluation(val_loader, model, attack, adv_test= True, jacobian_reg = True, lambda_JR = lambda_JR)
          super_loss_lst.append(loss_super_avg)
          JR_loss_lst.append(loss_JR_avg)
          #loss_lst.append(loss_avg)
        else:
          current_acc, loss_avg = val_evaluation(val_loader, model, attack, adv_test= False)
        

        if current_acc > best_acc:
          best_acc = current_acc
          if jacobian_reg:
            torch.save(model.state_dict(), path + 'JR_' + str(lambda_JR) + '_' + model_name)
          else:
            torch.save(model.state_dict(), path + 'standard_' + model_name)
        

        train_loss_avg /= num_samples
        train_loss_lst.append(train_loss_avg)
        val_loss_lst.append(loss_avg)
        

  epoch_lst = list(range(epochs))
  plt.plot(epoch_lst, train_loss_lst, label='train')
  plt.plot(epoch_lst, val_loss_lst, label='val')
  plt.xlabel('epoch')
  plt.ylabel('avg loss')
  plt.title('Loss curve')
  plt.legend()
  plt.show()

  
  if jacobian_reg:
    epoch_lst = list(range(epochs))
    plt.plot(epoch_lst, super_loss_lst, label='supervised loss')
    plt.plot(epoch_lst, JR_loss_lst, label='JR loss')
    plt.plot(epoch_lst, val_loss_lst, label='total loss')
    plt.xlabel('epoch')
    plt.ylabel('avg loss')
    plt.title('Validation loss curve')
    plt.legend()
    plt.show()

    plt.plot(epoch_lst, JR_loss_lst)
    plt.xlabel('epoch')
    plt.ylabel('avg loss')
    plt.title('JR loss curve')
    plt.show()
