#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:36:31 2021

@author: peter
"""

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image



class Segmentation_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, filetype = '.jpg', transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
    
        self.img_names = [x for x in Path(img_dir).glob(f'*{filetype}')]
        self.label_names = [x for x in Path(label_dir).glob(f'*{filetype}')]
        
        #need to make sure that the labels and image match - just use the numbers at the end of the 
        #filenames to sort the files
        img_order = [int(x.stem[-2:]) for x in self.img_names]
        label_order = [int(x.stem[-2:]) for x in self.label_names]
        
        self.img_names = [str(y[0]) for y in sorted(zip(self.img_names,img_order), key = lambda x: x[1])]
        self.label_names = [str(y[0]) for y in sorted(zip(self.label_names,label_order), key = lambda x: x[1])]
        
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self,idx):
        image = read_image(self.img_names[idx])
        label = read_image(self.label_names[idx])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

if __name__ == '__main__':
    dataset = Segmentation_Dataset('../2012/training_data/images','../2012/training_data/labels')
    tst_im, tst_lab = dataset.__getitem__(0)