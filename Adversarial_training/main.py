#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 19:37:42 2021

@author: chrischen
"""
import numpy as np
import PIL
from PIL import Image
import glob
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from train import train
from test_acc import check_accuracy
sys.path.append("..") 
from module_squeezenet import SqueezeNet
from Attack.PGD import PGD


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

def standard_model_training(train_loader, val_loader, test_loader, lr, gamma, step_size=1, epochs=30, path='../weights/'):
    
    model = SqueezeNet(version='1_1', num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    train_loss_lst, val_loss_lst = train(model, train_loader, val_loader, optimizer, scheduler, path, epochs=epochs, adv_train=False)
    model.load_state_dict(torch.load(path + 'standard_SqueezeNet.pt'))
    epsilon = 4/255
    alpha = 2/255 
    k = 10
    attack = PGD(model, 
            epsilon, 
            alpha, 
            min_val=-1, 
            max_val=1, 
            max_iters=k, 
            _type='linf')
    print('Test acc')
    check_accuracy(test_loader, model, attack, adv_test=False)
    print('Test acc adv ($\epsilon_{test}=4/255$)')
    check_accuracy(test_loader, model, attack, adv_test=True)

def adv_model_training(train_loader, val_loader, test_loader, epsilon, alpha, k, lr, gamma, step_size=1, epochs=30, path='../weights/'):
    
    model = SqueezeNet(version='1_1', num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    train_loss_lst, val_loss_lst = train(model, train_loader, val_loader, optimizer, scheduler, path, epsilon=epsilon, alpha=alpha, k=k, epochs=epochs, adv_train=True)
    model.load_state_dict(torch.load(path + 'adv_' + str(int(epsilon*255)) + '_SqueezeNet.pt'))
    epsilon_test = 4/255
    alpha_test = 2/255 
    k_test= 10
    attack = PGD(model, 
            epsilon_test, 
            alpha_test, 
            min_val=-1, 
            max_val=1, 
            max_iters=k_test, 
            _type='linf')
    print('Test acc')
    check_accuracy(test_loader, model, attack, adv_test=False)
    print('Test acc adv ($\epsilon_{test}=4/255$)')
    check_accuracy(test_loader, model, attack, adv_test=True)

def main():
    
    torch.manual_seed(6)
    custom_ad_dataset = CustomDatasetFromFile('../data/Ad_training_tensors/')
    #dataset_loader = DataLoader(custom_ad_dataset, batch_size=64, shuffle=True)
    train_dataset, temp_dataset = train_val_dataset(custom_ad_dataset, val_split=0.25)
    val_dataset, test_dataset = train_val_dataset(temp_dataset, val_split=0.5)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=True)
    
    '''
    #Standard training
    standard_lr = 9.28426850e-05
    standard_gamma = 0.90
    standard_model_training(train_loader, val_loader, test_loader, lr=standard_lr, gamma=standard_gamma)
    '''
    
    #Adversarial training
    adv_lr = 0.000375
    adv_gamma = 0.99
    epsilon_train = 4/255
    alpha_train = epsilon_train/2
    k_train = 10
    adv_model_training(train_loader, val_loader, test_loader, epsilon=epsilon_train, alpha=alpha_train, k=k_train, lr=adv_lr, gamma=adv_gamma, epochs=30)
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
