#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:42:48 2021

@author: peter
"""
import torch
import torch.nn

import numpy as np

from . import unet_parts

class Unet(torch.nn.Module):
    '''
     Reproducing the Unet model from https://arxiv.org/pdf/1505.04597v1.pdf
    '''
    
    def __init__(self):
        super(Unet,self).__init__()
        
        
        #Define the convolutional layers for the encoding
        channels_out  = [64*2**x for x in range(5)]
        channels_in   = [1] + channels_out[:-1]
        self.encoder_convs = torch.nn.ModuleList([unet_parts.double_conv(*x) for x in zip(channels_in,channels_out)])
        self.max_pool_2x2 = torch.nn.MaxPool2d(kernel_size = 2,stride = 2)
        

        #define the upsampling transpose convolutions so use the reverse channels
        self.up_convs = torch.nn.ModuleList([torch.nn.ConvTranspose2d(*x, kernel_size = 2, stride = 2, padding = 0) for x in zip(channels_out,channels_in)][:0:-1])
        #The double convs are on the concatenated data with the passover connections
        #so have the same number of channels
        self.decoder_convs = torch.nn.ModuleList([unet_parts.double_conv(*x) for x in zip(channels_out,channels_in)][:0:-1])
        
        self.output_conv = torch.nn.Conv2d(64, 2, kernel_size =  1, padding = 0)
        self.sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self,image):
        '''
        Implemented exactly as in Unet paper
        We iteratively encode then decode the image before 

        '''
        self.crops = unet_parts.calculate_crops(image.shape[-1])
        
        #first we encode the image
        self.intermediates = [image]
        for idx,encoder_conv_layer in enumerate(self.encoder_convs):
            #encode with double conv layers
            self.intermediates.append(encoder_conv_layer(self.intermediates[-1]))
            #Don't max pool the final layer
            if idx != len(self.encoder_convs) -1:
                self.intermediates.append(self.max_pool_2x2(self.intermediates[-1]))
        
        #We don't actually need to save these intermediate layers now as they are
        #not used, but they aree useful for debugging so for now we will
        for idx, (crop, up_conv_layer, decoder_conv_layer) in enumerate(zip(self.crops, self.up_convs,self.decoder_convs)):
            #upsample and concatenate with encoded result for passover connections
            self.intermediates.append(torch.cat([self.intermediates[7 - 2*idx][...,crop:-crop,crop:-crop], 
                                                 up_conv_layer(self.intermediates[-1])],dim = 1))
            #reduce channels with decoder
            self.intermediates.append(decoder_conv_layer(self.intermediates[-1]))

        
        #finally use 1x1 conv sigmoid activation to map to seg
        out = self.sigmoid(self.output_conv(self.intermediates[-1]))
        
        return out
        
    def pretraining_initialise(self):
        self.apply(unet_parts.init_normal_weights)
        
if __name__ == '__main__':
    image = torch.rand(1,1,572,572)
    model = Unet()
    print(model(image))
