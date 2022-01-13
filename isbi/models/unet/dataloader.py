#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:36:31 2021

@author: peter
"""

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as torchtransforms
import torchvision.transforms.functional as TF
import numpy as np

import torch


class Segmentation_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, filetype=".jpg", augmentation=1000):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augmentation = augmentation

        self.img_names = [x for x in Path(img_dir).glob(f"*{filetype}")]
        self.label_names = [x for x in Path(label_dir).glob(f"*{filetype}")]

        # need to make sure that the labels and image match
        # just use the numbers at the end of the
        # filenames to sort the files
        img_order = [int(x.stem[-2:]) for x in self.img_names]
        label_order = [int(x.stem[-2:]) for x in self.label_names]

        self.img_names = [
            str(y[0])
            for y in sorted(zip(self.img_names, img_order), key=lambda x: x[1])
        ]
        self.label_names = [
            str(y[0])
            for y in sorted(zip(self.label_names, label_order), key=lambda x: x[1])
        ]

        # assumes augmntation >>1 so efficient to read in data first
        self.images = [read_image(x) for x in self.img_names]
        self.labels = [read_image(x) for x in self.label_names]

    def __len__(self):
        return len(self.img_names) * self.augmentation

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        else:
            aug_idx, im_idx = np.divmod(idx, len(self.img_names))

            image = self.images[im_idx]
            label = self.labels[im_idx]

            # don't augment once per epoch
            if aug_idx != 0:
                image, label = self.transform(image, label)
            else:
                image = torchtransforms.CenterCrop(508)(image)
                label = torchtransforms.CenterCrop(508)(label)

            return self.shape_im(image), self.shape_labels(label)

    def transform(self, image, mask):
        # functional style to apply same random transforms to label and image
        affine_params = torchtransforms.RandomAffine.get_params(
            (-180, 180),
            translate=None,
            scale_ranges=(0.8, 1.2),
            shears=(-10, 10, -10, 10),
            img_size=[512, 512],
        )

        image = TF.affine(image, *affine_params)
        mask = TF.affine(mask, *affine_params)

        # Random horizontal flipping
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if np.random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        crop_params = torchtransforms.RandomCrop.get_params(
            image, output_size=(508, 508)
        )
        image = TF.crop(image, *crop_params)
        mask = TF.crop(mask, *crop_params)

        return image, mask

    def shape_im(self, image):
        return image.reshape((1,) + image.shape) / 255

    def shape_labels(self, labels):
        labels = torchtransforms.CenterCrop(324)(labels)
        return np.round((labels / 255)).type(torch.long)


if __name__ == "__main__":
    dataset = Segmentation_Dataset(
        "../2012/training_data/images", "../2012/training_data/labels"
    )
    tst_im, tst_lab = dataset.__getitem__(0)
