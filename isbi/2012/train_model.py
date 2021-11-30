#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:12:47 2021

@author: peter
"""

import torch
import torch.nn
import torchvision.transforms as torchtransforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from isbi.models import unet,dataloader
from isbi.util import plotting

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


model = unet.Unet().to(device)
model.pretraining_initialise()

training_data = dataloader.Segmentation_Dataset('./training_data/images',
                                                './training_data/labels')

im,labels = training_data.__getitem__(5001)
    
plotting.show_labels(im,labels)


loss_fn = torch.nn.CrossEntropyLoss()
def calculate_loss(loss_fn, output, labels):
    return loss_fn(output.reshape((2,-1)).T, labels.reshape(-1))


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_data):
        # get the inputs; data is a list of [inputs, labels]
        im, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(im)
        loss = calculate_loss(loss_fn, output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
        break
    break
        

print('Finished Training')

print(model)