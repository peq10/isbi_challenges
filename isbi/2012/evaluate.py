#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:40:06 2021

@author: quickep
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from isbi.models import unet, dataloader

from isbi.util import plotting

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


model = unet.Unet().to(device)
model.load_state_dict(
    torch.load("./training_data/checkpoint_epoch5.nn", map_location=torch.device("cpu"))
)
model.eval()

training_data = dataloader.Segmentation_Dataset(
    "./training_data/images", "./training_data/labels"
)

im, labels = training_data.__getitem__(0)


def show_labels_res(im, labels, alpha=0.5):
    # just checks the labels by overlaying on imag
    im_sh = np.array(im.shape[-2:])
    lab_sh = np.array(labels.shape[-2:])
    diff_sh = (im_sh - lab_sh) / 2
    b, a = np.floor(diff_sh).astype(int), np.ceil(diff_sh).astype(int)

    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(im), cmap="Greys_r")

    labels = np.pad(np.squeeze(labels), ((b[0], a[0]), (b[1], a[1])))
    overlay = labels[..., None] * np.array([255, 0, 0, int(255 * alpha)])
    ax.imshow(overlay)


show_labels_res(im, labels)

# get the inputs; data is a list of [inputs, labels]
im, labels = im.to(device), labels.to(device)

# forward + backward + optimize
output = model(im)
probs = torch.nn.Softmax(dim=1)(output)


classes = np.argmax(probs.detach().numpy(), 1).squeeze()

plotting.show_error(labels.numpy(), classes, alpha=0.8)
