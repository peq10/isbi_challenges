import numpy as np

import torch.nn


def double_conv(channels_in, channels_out):
    """
    Generates the double convolutional layers for unet

    """
    layers = torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=0),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=0),
        torch.nn.ReLU(inplace=True),
    )
    return layers


def init_normal_weights(layer):
    name = layer.__class__.__name__
    if name.find("Linear") != -1:
        y = layer.in_features
        # see unet paper for reasoning
        layer.weight.data.normal_(0.0, np.sqrt(2 / y))
        layer.bias.data.fill_(0)


def get_valid_size(image_size):
    # returns the largest integer for which the unet will work
    # this is all due to the unpadded convolutions and pooling
    print("hello")
    for i in range(image_size - image_size % 2, 2, -2):
        try:
            calculate_crops(i, recurse=False)
            return i
        except Exception:
            continue


def calculate_crops(image_size, recurse=True):
    # calculates what crops we need for the concatenation in the decoder
    # this is due to the unpadded convolutions and pooling
    # slightly hacky think to stop recursion
    down_sizes = np.zeros(5, dtype=int)
    sz = image_size
    for i in range(5):
        sz = int(sz) - 4
        down_sizes[i] = sz
        sz /= 2

    up_sizes = np.zeros_like(down_sizes)

    sz *= 2
    for i in range(5):
        sz *= 2
        up_sizes[i] = sz
        sz -= 4

    crops = (down_sizes[-2::-1] - up_sizes[:-1]) / 2
    if any(crops % 2 != 0):
        # hacky way to do this - refactor
        if recurse:
            closest_valid_size = get_valid_size(image_size)
        else:
            closest_valid_size = None
        raise ValueError(f"Image wrong size. crop to {closest_valid_size}")

    return crops.astype(int)
