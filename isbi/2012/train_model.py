#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:12:47 2021

@author: peter
"""
import logging
import torch
import torch.nn
import wandb
import sys

import torch.optim as optim

from pathlib import Path
from tqdm import tqdm

from isbi.models import unet, dataloader


def train_unet(
    unet,
    device,
    epochs=5,
    batch_size=1,
    learning_rate=0.001,
    validation_frac=0.1,
    class_weights=None,
    training_directory="./training_data/",
    augmentation=1,
    use_wandb=False,
):

    # get dataset
    dataset = dataloader.Segmentation_Dataset(
        Path(training_directory, "images"),
        Path(training_directory, "labels"),
        augmentation=augmentation,
    )

    # get train and val partitions
    # TODO

    # log data
    if use_wandb:
        experiment = wandb.init(project="Unet", resume="allow", anonymous="must")
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_frac=validation_frac,
                class_weights=class_weights,
            )
        )
    checkpoint_dir = Path(training_directory, "checkpoints")

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Class weights: {class_weights}
    """
    )

    # get the optimizer and loss
    optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    def calculate_loss(loss_fn, output, labels):
        return loss_fn(output.reshape((2, -1)).T, labels.reshape(-1))

    global_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(
            total=len(dataset), desc=f"Epoch {epoch + 1}/{epochs}", unit=" img"
        ) as pbar:
            for i, data in enumerate(dataset):
                # get the inputs; data is a list of [inputs, labels]
                im, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = unet(im)
                loss = calculate_loss(loss_fn, output, labels)
                loss.backward()
                optimizer.step()

                # log statistics
                running_loss += loss.item()
                global_step += 1

                pbar.update(i)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                if use_wandb:
                    experiment.log(
                        {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                    )

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            unet.state_dict(), Path(checkpoint_dir, f"checkpoint_epoch{epoch + 1}.nn"),
        )
        logging.info(f"Checkpoint {epoch + 1} saved!")

    print("Finished Training")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = unet.Unet()
    model.pretraining_initialise()
    model.to(device)

    training_dir = "./training_data/"

    try:
        train_unet(
            model,
            device,
            epochs=5,
            batch_size=1,
            learning_rate=0.001,
            validation_frac=0.1,
            class_weights=torch.tensor([0.8, 0.2]),
            training_directory=training_dir,
            augmentation=1000,
            use_wandb=False,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), Path(training_dir, "INTERRUPTED.nn"))
        logging.info("Saved interrupt")
        sys.exit(0)
