# Imports
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.models as models
from torchvision import transforms, utils,datasets
import torch.utils.data as data
from torch.optim import lr_scheduler
from PIL import Image
import os
import os.path
from torch.utils.data import Dataset
from tqdm import tqdm as tqdmn
import torch.nn as nn
import skimage
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from decimal import Decimal
import shutil

def load_checkpoint(model_best_path, model=None, optimizer=None, scheduler=None, epoch = 0, evals = (None, None)):
    # Load best model
    if os.path.isfile(model_best_path):
        # Load states
        checkpoint = torch.load(model_best_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Update settings
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        r2_scores = checkpoint['r2_scores']
        evals = (losses, r2_scores)
        print("Loaded checkpoint '{}' (epoch {} successfully.".format(model_best_path, epoch))
    else:
        print("No checkpoint found.")
    return model, optimizer, epoch, evals

def save_checkpoint(state, is_best, filename, checkpoint_dir):
    """Saves latest model
    Parameters
    ----------
    state : dict
        State of the model to be saved
    is_best : boolean
        Whether or not current model is the best model
    checkpoint_dir : str
        Path to models
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_dir + filename)
    if is_best:
        shutil.copyfile(
            checkpoint_dir + filename,
            checkpoint_dir + "model_best.pt",
        )

def train(model, train_loader, val_loader, criterion, optimizer, chosen_metric, scheduler, device, checkpoint_dir, writer,
          best_metric = -np.inf, num_epochs=5, curr_epoch=0, save_epoch=50, multiple_crops=False):
    """ Trains the model  and outputs best optimized model with full best state"""
    #
    phases = ["train", "val"]
    dataloaders = {"train": train_loader, "val": val_loader}
    losses = {phase: [] for phase in phases}
    metric_scores = {phase: [] for phase in phases}
    # Go through dataset
    for epoch in range(curr_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            preds_ = []
            labels_ = []
            # Iterate over data.
            for idx, (batch, label) in tqdmn(enumerate(dataloaders[phase])):
                if multiple_crops:
                    bs, ncrops, c, h, w = batch.size()
                    batch = batch.view(-1, c, h, w)
                batch, label = batch.to(device), label.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward: track history if only in train
                outputs = model(batch)
                if multiple_crops:
                    outputs = outputs.view(bs, ncrops, -1).mean(1)
                loss = criterion(outputs.type(torch.cuda.FloatTensor), label.type(torch.cuda.FloatTensor))
                preds_.extend(outputs.cpu().detach().numpy().tolist())
                labels_.extend(label.data.cpu().numpy().tolist())
                running_loss += loss.item() * batch.size(0)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # Out of current epoch
            epoch_loss = running_loss / len(dataloaders[phase].sampler)
            #epoch_metrics = chosen_metric(labels_,np.argmax(preds_,axis=1))
            epoch_metrics = chosen_metric(labels_,preds_)
            losses[phase].append(epoch_loss)
            metric_scores[phase].append(epoch_metrics)
            learning_rate = optimizer.param_groups[0]["lr"]
            #
            print(
                "{} Loss: {:.4f} Metric Score: {:.4f} LR: {:.4E}".format(
                    phase.upper(),
                    epoch_loss,
                    epoch_metrics,
                    Decimal(learning_rate),)
            )
            if phase == "val":
                # Update scheduler
                scheduler.step()
                # Check if current model gives the best F1 score
                is_best = False
                if epoch_metrics > best_metric:
                    best_metric = epoch_metrics
                    is_best = True
                    best_state = {"epoch": epoch + 1,
                             "lr": learning_rate,
                             "state_dict": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "losses": losses,
                             "metric_scores": metric_scores}
            writer.add_scalar(phase + '/Loss', epoch_loss, epoch)
            writer.add_scalar(phase + '/Metric', epoch_metrics, epoch)
        #
        if learning_rate <= 1e-9:
            break
        #
        if epoch % save_epoch == 0:
            save_checkpoint(best_state, is_best,"model_epoch_%d.ckpt.tar"%epoch, checkpoint_dir)
    #
    print("Best Val R2: {:4f}".format(best_metric))
    model.load_state_dict(best_state["state_dict"])
    return model, best_state

def test(test_loader, model, criterion, metric, device, multiple_crops=False):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        preds_ = []
        labels_ = []
        for idx, (batch, label) in tqdmn(enumerate(test_loader)):
            if multiple_crops:
                bs, ncrops, c, h, w = batch.size()
                batch = batch.view(-1, c, h, w)
            batch, label = batch.to(device), label.to(device)
            # compute output
            outputs = model(batch)
            if multiple_crops:
                outputs = outputs.view(bs, ncrops, -1).mean(1)
            loss = criterion(outputs.type(torch.cuda.FloatTensor), label.type(torch.cuda.FloatTensor))
            preds_.extend(outputs.cpu().detach().numpy().tolist())
            labels_.extend(label.data.cpu().numpy().tolist())
        #test_metrics = chosen_metric(labels_,np.argmax(preds_,axis=1))
        test_metrics = chosen_metric(labels_,preds_)
        print("Test Metrics: {:4f}".format(test_metrics))
    return test_metrics
