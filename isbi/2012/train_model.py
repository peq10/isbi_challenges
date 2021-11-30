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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


model = unet.Unet()
model.pretraining_initialise()
model.to(device)

training_data = dataloader.Segmentation_Dataset('./training_data/images',
                                                './training_data/labels')


def show_labels(im,labels, alpha = 0.5):
    #just checks the labels by overlaying on imag
    im_sh = np.array(im.shape[-2:])
    lab_sh = np.array(labels.shape[-2:])
    diff_sh = (im_sh - lab_sh)/2
    b,a = np.floor(diff_sh).astype(int), np.ceil(diff_sh).astype(int)
    
    fig,ax = plt.subplots()
    ax.imshow(np.squeeze(im),cmap = 'Greys_r')
    
    labels = np.pad(np.squeeze(labels),((b[0],a[0]),(b[1],a[1])))
    overlay = labels[...,None]*np.array([255,0,0,int(255*alpha)])
    ax.imshow(overlay)
    

loss_fn = torch.nn.CrossEntropyLoss()
def calculate_loss(loss_fn, output, labels):
    return loss_fn(output.reshape((2,-1)).T, labels.reshape(-1))


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(len(training_data))
losses = []
model.cuda()
for epoch in range(5):  # loop over the dataset multiple times

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

        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
            losses.append((epoch, i, loss.item()))

        if i > 0 and i % 20000 == 0:
            torch.save(model.state_dict(), './training_data/trained_model2.nn')

            np.savetxt('./training_data/losses.txt', losses)
            break
 

print('Finished Training')

print(model)

torch.save(model.state_dict(), './training_data/trained_model.nn')

np.savetxt('./training_data/losses.txt', losses)