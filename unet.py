#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:42:48 2021

@author: peter
"""
import torch
import torch.nn


def double_conv(channels_in,channels_out):
    '''
    Generates the double convolutional layers for unet

    '''
    layers = torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding = 'valid'),
        torch.nn.ReLU(inplace = True),
        torch.nn.Conv2d(channels_out, channels_out, kernel_size = 3, padding = 'valid'),
        torch.nn.ReLU(inplace = True)
        )
    return layers

class Unet(torch.nn.Module):
    '''
     Reproducing the Unet model from https://arxiv.org/pdf/1505.04597v1.pdf
    '''
    
    def __init__(self):
        super(Unet,self).__init__()
        
        
        #Define the convolutional layers for the encoding
        channels_out  = [64*2**x for x in range(5)]
        channels_in   = [1] + channels_out[:-1]
        self.encoder_convs = [double_conv(*x) for x in zip(channels_in,channels_out)]
        self.max_pool_2x2 = torch.nn.MaxPool2d(kernel_size = 2,stride = 2)
        

        #define the upsampling transpose convolutions so use the reverse channels
        self.up_convs = [torch.nn.ConvTranspose2d(*x, kernel_size = 2, stride = 2, padding = 0) for x in zip(channels_out,channels_in)][:0:-1]
        #The double convs are on the concatenated data with the passover connections
        #so have the same number of channels
        self.decoder_convs = [double_conv(*x) for x in zip(channels_out,channels_in)][:0:-1]
        
        #A helper to tell us how much to crop when concatenating on the decoder
        self.crops = [4,16,40,88]
        
        self.output_conv = torch.nn.Conv2d(64, 2, kernel_size =  1, padding = 'valid')
        self.sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self,image):
        '''
        Implemented exactly as in Unet paper
        We iteratively encode then decode the image before 

        '''
        
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

        for idx, crop, up_conv_layer, decoder_conv_layer in zip(range(5), self.crops, self.up_convs,self.decoder_convs):
            #upsample and concatenate with encoded result for passover connections
            self.intermediates.append(torch.cat([self.intermediates[7 - 2*idx][...,crop:-crop,crop:-crop], 
                                                 up_conv_layer(self.intermediates[-1])],dim = 1))
            #reduce channels with decoder
            self.intermediates.append(decoder_conv_layer(self.intermediates[-1]))

        
        #finally use 1x1 conv sigmoid activation to map to seg
        out = self.sigmoid(self.output_conv(self.intermediates[-1]))
        
        print(out.shape)
        return out
        
        
if __name__ == '__main__':
    image = torch.rand(1,1,572,572)
    model = Unet()
    print(model(image))