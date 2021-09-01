#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 21:19:52 2021

@author: chrischen
"""
import glob
import sys
import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from utils import check_accuracy, check_acc_rand, check_acc_rand_and_combined
sys.path.append("..") 
from module_squeezenet import SqueezeNet
from Attack.PGD import PGD

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.image_list = glob.glob(folder_path+'*')
        # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        '''
        img = Image.open(single_image_path)
        img = img.convert('RGB')

        # Do some operations on image
        # input dimension = 3x224x224
        h,w = 224,224
        width, height = img.size   # Get dimensions
    
        #if width < 100 or height < 100:
            #return None
        
        if width == 225 and height == 225:
            left = 0
            top = 0
            right = w
            bottom = h
            img = img.crop((left, top, right, bottom))
        else:
            img = img.resize((h, w), PIL.Image.ANTIALIAS)

        img = np.asarray(img).astype(np.float32)
        img = (img / 127.5) - 1.0
        


        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(np.transpose(img, (2,0,1))).float()
        '''

        im_as_ten = torch.load(single_image_path)
        '''
        img = cv2.imread(single_image_path,cv2.IMREAD_UNCHANGED)
        height,width,channel = img.shape
        h,w = 224,224
        if width == 225 and height == 225:
          img = img[0:h,0:w]
        else:
          img = cv2.resize(img,(h,w))

        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        im_as_ten = torch.from_numpy(np.transpose(img, (2,0,1))).float()
        '''

        # Get label(class) of the image based on the file name
        if 'Positive' in single_image_path:
          label = 1.0
        else:
          label = 0.0

        return (im_as_ten, label)

    def __len__(self):
        return self.data_len
    
    
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train = Subset(dataset, train_idx)
    val = Subset(dataset, val_idx)
    return train, val


def main():
    
    torch.manual_seed(6)
    custom_ad_dataset = CustomDatasetFromFile('../data/Ad_training_tensors/')
    #dataset_loader = DataLoader(custom_ad_dataset, batch_size=64, shuffle=True)
    train_dataset, temp_dataset = train_val_dataset(custom_ad_dataset, val_split=0.25)
    val_dataset, test_dataset = train_val_dataset(temp_dataset, val_split=0.5)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=True)
    
    path = '../weights/'
    if not os.path.exists('results'):
        os.makedirs('results')
    #Randomized JR
    ind_str = ['1', '2']
    lambda_JR = 15
    model_lst = []
    model_orig = SqueezeNet(version='1_1', num_classes=2)
    model_path_orig = path + 'JR_' + str(lambda_JR) + '_' + 'SqueezeNet.pt'
    model_orig.load_state_dict(torch.load(model_path_orig))
    model_orig = model_orig.to(device=device)
    model_lst.append(model_orig)
    for index in ind_str:
      model = SqueezeNet(version='1_1', num_classes=2)
      model_name = index + '_SqueezeNet.pt'
      model_path = path + 'JR_' + str(lambda_JR) + '_' + model_name
      model.load_state_dict(torch.load(model_path))
      model = model.to(device=device)
      model_lst.append(model)
    
    diff_model_lst = []
    lambda_lst = ['10', '15_1', '20']
    for lam in lambda_lst:
      model = SqueezeNet(version='1_1', num_classes=2)
      model_name = 'SqueezeNet.pt'
      model_path = path + 'JR_' + lam + '_' + model_name
      model.load_state_dict(torch.load(model_path))
      model = model.to(device=device)
      diff_model_lst.append(model)
    
    
    epsilon_x_axis = np.array(list(range(17)))*1/255
    x_values = []
    for i in range(17):
      x_values.append('%d'%(i))
    
    
    attacked_model = diff_model_lst[1]
    rand_acc_lst = []
    rand_acc_lst_diff = []
    sing_acc_lst = []
    sing_acc_lst_10 = []
    sing_acc_lst_20 = []
    epsilon_list = list(range(17))
    for epsilon in epsilon_list:
        print('Adv test acc (epsilon=%d/255)'%(epsilon))
        epsilon = epsilon * 1/255
        alpha = epsilon/2
        attack = PGD(attacked_model, 
                    epsilon, 
                    alpha, 
                    min_val=-1, 
                    max_val=1, 
                    max_iters=10, 
                    _type='linf')
        
        attack_10 = PGD(diff_model_lst[0], 
                        epsilon, 
                        alpha, 
                        min_val=-1, 
                        max_val=1, 
                        max_iters=10, 
                        _type='linf')
        
        attack_20 = PGD(diff_model_lst[2], 
                        epsilon, 
                        alpha, 
                        min_val=-1, 
                        max_val=1, 
                        max_iters=10, 
                        _type='linf')
        
        attack_15 = PGD(model_lst[2], 
                        epsilon, 
                        alpha, 
                        min_val=-1, 
                        max_val=1, 
                        max_iters=10, 
                        _type='linf')
        
        acc_random = check_acc_rand(test_loader, model_lst, attack, adv_test=True)
        acc_random_diff = check_acc_rand(test_loader, diff_model_lst, attack, adv_test=True)
        acc_sing = check_accuracy(test_loader, model_lst[2], attack_15, adv_test=True)
        acc_10 = check_accuracy(test_loader, diff_model_lst[0], attack_10, adv_test=True)
        acc_20 = check_accuracy(test_loader, diff_model_lst[2], attack_20, adv_test=True)
        rand_acc_lst.append(acc_random)
        rand_acc_lst_diff.append(acc_random_diff)
        sing_acc_lst.append(acc_sing)
        sing_acc_lst_10.append(acc_10)
        sing_acc_lst_20.append(acc_20)
    plt.plot(epsilon_x_axis, rand_acc_lst, label='Randomized(same)')
    plt.plot(epsilon_x_axis, rand_acc_lst_diff, label='Randomized(diff)')
    plt.plot(epsilon_x_axis, sing_acc_lst, label='Individual(λ=15)')
    plt.plot(epsilon_x_axis, sing_acc_lst_10, label='Individual(λ=10)')
    plt.plot(epsilon_x_axis, sing_acc_lst_20, label='Individual(λ=20)')
    
    
    plt.xlabel('$ϵ_{test}$(1/255)')
    plt.xticks(epsilon_x_axis,x_values)
    plt.ylabel('accuracy')
    plt.title('Comparison of Randomized and Individual models')
    plt.legend()
    plt.savefig('results/JR_randomized_individual.png')
    
    
    #Randomized adversarial training
    ind_str = ['1', '2']
    eps = 5/255
    model_lst = []
    model_orig = SqueezeNet(version='1_1', num_classes=2)
    model_path_orig = path + 'adv_' + str(int(eps*255)) + '_SqueezeNet.pt'
    model_orig.load_state_dict(torch.load(model_path_orig))
    model_orig = model_orig.to(device=device)
    model_lst.append(model_orig)
    for index in ind_str:
      model = SqueezeNet(version='1_1', num_classes=2)
      model_name = index + '_SqueezeNet.pt'
      model_path = path + 'adv_' + str(int(eps*255)) + '_' + model_name
      model.load_state_dict(torch.load(model_path))
      model = model.to(device=device)
      model_lst.append(model)
    
    model_lst_diff = []
    eps_lst = ['3', '4', '5_2']
    for epsilon in eps_lst:
      model = SqueezeNet(version='1_1', num_classes=2)
      model_name = 'SqueezeNet.pt'
      model_path = path + 'adv_' + epsilon + '_' + model_name
      model.load_state_dict(torch.load(model_path))
      model = model.to(device=device)
      model_lst_diff.append(model)
    
    print(len(model_lst))
    
    epsilon_x_axis = np.array(list(range(17)))*1/255
    x_values = []
    for i in range(17):
      x_values.append('%d'%(i))
    
    
    attacked_model = model_lst_diff[2]
    rand_acc_lst_adv_train = []
    rand_acc_lst_diff = []
    sing_acc_lst = []
    sing_acc_lst_3 = []
    sing_acc_lst_4 = []
    epsilon_list = list(range(17))
    for epsilon in epsilon_list:
        print('Adv test acc (epsilon=%d/255)'%(epsilon))
        epsilon = epsilon * 1/255
        alpha = epsilon/2
        attack = PGD(attacked_model, 
                    epsilon, 
                    alpha, 
                    min_val=-1, 
                    max_val=1, 
                    max_iters=10, 
                    _type='linf')
        
        attack_3 = PGD(model_lst_diff[0], 
                        epsilon, 
                        alpha, 
                        min_val=-1, 
                        max_val=1, 
                        max_iters=10, 
                        _type='linf')
        
        attack_4 = PGD(model_lst_diff[1], 
                    epsilon, 
                    alpha, 
                    min_val=-1, 
                    max_val=1, 
                    max_iters=10, 
                    _type='linf')
        
        
        acc_random = check_acc_rand(test_loader, model_lst, attack, adv_test=True)
        acc_random_diff = check_acc_rand(test_loader, model_lst_diff, attack, adv_test=True)
        acc_sing = check_accuracy(test_loader, attacked_model, attack, adv_test=True)
        acc_sing_3 = check_accuracy(test_loader, model_lst_diff[0], attack_3, adv_test=True)
        acc_sing_4 = check_accuracy(test_loader, model_lst_diff[1], attack_4, adv_test=True)
        rand_acc_lst_adv_train.append(acc_random)
        rand_acc_lst_diff.append(acc_random_diff)
        sing_acc_lst.append(acc_sing)
        sing_acc_lst_3.append(acc_sing_3)
        sing_acc_lst_4.append(acc_sing_4)
    plt.plot(epsilon_x_axis, rand_acc_lst_adv_train, label='Randomized(same)')
    plt.plot(epsilon_x_axis, rand_acc_lst_diff, label='Randomized(diff)')
    plt.plot(epsilon_x_axis, sing_acc_lst, label='Individual($ϵ_{train}$=5/255)')
    plt.plot(epsilon_x_axis, sing_acc_lst_3, label='Individual($ϵ_{train}$=3/255)')
    plt.plot(epsilon_x_axis, sing_acc_lst_4, label='Individual($ϵ_{train}$=4/255)')
    
    
    plt.xlabel('$ϵ_{test}$(1/255)')
    plt.xticks(epsilon_x_axis,x_values)
    plt.ylabel('accuracy')
    plt.title('Comparison of Randomized and Individual models')
    plt.legend()
    plt.savefig('results/advtrain_randomized_individual.png')
    
    #Ensemble and Randomized
    ind_str = ['1', '2']
    lambda_JR = 15
    eps = 5/255
    JR_model_lst = []
    adv_model_lst = []
    
    model_JR_orig = SqueezeNet(version='1_1', num_classes=2)
    model_JR_path_orig = path + 'JR_' + str(lambda_JR) + '_' + 'SqueezeNet.pt'
    model_JR_orig.load_state_dict(torch.load(model_JR_path_orig))
    model_JR_orig = model_JR_orig.to(device=device)
    JR_model_lst.append(model_JR_orig)
    
    for index in ind_str:
      model_JR = SqueezeNet(version='1_1', num_classes=2)
      model_name = index + '_SqueezeNet.pt'
      model_JR_path = path + 'JR_' + str(lambda_JR) + '_' + model_name
      model_JR.load_state_dict(torch.load(model_JR_path))
      model_JR = model_JR.to(device=device)
      JR_model_lst.append(model_JR)
    
    eps_lst = ['3', '4', '5_2']
    for epsilon in eps_lst:
      model = SqueezeNet(version='1_1', num_classes=2)
      model_name = 'SqueezeNet.pt'
      model_path = path + 'adv_' + epsilon + '_' + model_name
      model.load_state_dict(torch.load(model_path))
      model = model.to(device=device)
      adv_model_lst.append(model)
    
    epsilon_x_axis = np.array(list(range(17)))*1/255
    x_values = []
    for i in range(17):
      x_values.append('%d'%(i))
    
    
    attacked_model = JR_model_lst[1]
    combined_acc_lst = []
    rand_JR_acc_lst = []
    rand_adv_acc_lst = []
    JR_acc_lst = []
    adv_acc_lst = []
    epsilon_list = list(range(17))
    for epsilon in epsilon_list:
        print('Adv test acc (epsilon=%d/255)'%(epsilon))
        epsilon = epsilon * 1/255
        alpha = epsilon/2
        
        attack = PGD(attacked_model, 
                    epsilon, 
                    alpha, 
                    min_val=-1, 
                    max_val=1, 
                    max_iters=10, 
                    _type='linf')
        attack_adv = PGD(adv_model_lst[2], 
                        epsilon, 
                        alpha, 
                        min_val=-1, 
                        max_val=1, 
                        max_iters=10, 
                        _type='linf')
    
        
        acc_combined = check_acc_rand_and_combined(test_loader, adv_model_lst, JR_model_lst, attack, adv_test=True)
        acc_rand_JR = check_acc_rand(test_loader, JR_model_lst, attack, adv_test=True)
        acc_rand_adv = check_acc_rand(test_loader, adv_model_lst, attack_adv, adv_test=True)
        acc_JR = check_accuracy(test_loader, attacked_model, attack, adv_test=True)
        acc_adv = check_accuracy(test_loader, adv_model_lst[2], attack_adv, adv_test=True)
        combined_acc_lst.append(acc_combined)
        rand_JR_acc_lst.append(acc_rand_JR)
        rand_adv_acc_lst.append(acc_rand_adv)
        JR_acc_lst.append(acc_JR)
        adv_acc_lst.append(acc_adv)
    
    plt.plot(epsilon_x_axis, combined_acc_lst, label='Ensemble & Randomized', linewidth=4, linestyle=(0, (5, 2, 1, 2)))
    plt.plot(epsilon_x_axis, rand_JR_acc_lst, label='JR(Randomized)')
    plt.plot(epsilon_x_axis, rand_adv_acc_lst, label='Adv_train(Randomized)')
    plt.plot(epsilon_x_axis, JR_acc_lst, label='JR(λ=15)')
    plt.plot(epsilon_x_axis, adv_acc_lst, label='Adv_train($ϵ_{train}$=5)')
    
    
    plt.xlabel('$ϵ_{test}$(1/255)')
    plt.xticks(epsilon_x_axis,x_values)
    plt.ylabel('accuracy')
    plt.title('Comparison of models')
    plt.legend()
    plt.savefig('results/JR_ensemble_randomized.png')

