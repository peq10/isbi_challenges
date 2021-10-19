#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:12:47 2021

@author: peter
"""

import torch
import torch.nn
import torchvision.transforms as torchtransforms

from isbi.models import unet,dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def transform_im(im):
    return torchtransforms.CenterCrop(508)((im.reshape((1,) + im.shape))/255)


def transform_target(lab):
    lab = torchtransforms.CenterCrop(324)(torchtransforms.CenterCrop(508)(lab))
    return torch.stack([lab == 0, lab == 255],axis = 1).type(torch.float16)
     



model = unet.Unet().to(device)

training_data = dataloader.Segmentation_Dataset('./training_data/images',
                                                './training_data/labels',
                                                transform = transform_im,
                                                target_transform = transform_target)
                                                
tst_im,tst_label = training_data.__getitem__(0)

tst_output = model(tst_im)

loss_fn = torch.nn.CrossEntropyLoss()
print(model)