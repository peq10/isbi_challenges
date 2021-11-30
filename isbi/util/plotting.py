import matplotlib.pyplot as plt

import numpy as np


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