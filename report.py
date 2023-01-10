import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch import nn
from torchvision.transforms import ToTensor, Lambda, ToPILImage, RandomRotation
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from my_utils import saving_model, plot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        
def export_figure(log):
    pair1 = ['train_loss', 'train_acc']
    pair2 = ['train_loss', 'val_loss']
    pair3 = ['val_acc', 'val_loss']
    pair4 = ['val_acc', 'train_acc']
    for pair in [pair1, pair2, pair3, pair4]:
        fig_name = ' vs '.join(pair)
        plot(log, pair, fig_name)
        
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    log_idx = 100
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):        
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        size_batch = len(y)
        pred = model(X)
        loss = loss_fn(pred, y)
        acc = (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % log_idx == 0:
            loss, current = loss.item(), batch * len(y)
            print(f"Loss: {loss:>0.3f}| Acc: {(acc * 100 / size_batch):>0.3f}%  [{current:>5d}/{size:>5d}]")
        
        train_loss += loss
        train_acc += acc
    return torch.mean(train_loss).item(), train_acc * 100 / size
            
def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    val_loss, val_acc = 0, 0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device, dtype=torch.float), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    val_loss /= size
    val_acc /= size
    print(f"Test Error: \n Accuracy: {(100 * val_acc):>0.3f}%, Avg loss: {val_loss:>4f} \n")
    return val_loss, val_acc * 100

def report(model, model_name, dataloaders, learning_rate, epoch, L2=None):
    # Initialize the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = None
    if L2:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_params = {'dataloader': dataloaders[0],
                    'model': model,
                    'loss_fn': loss_fn,
                    'optimizer': optimizer}

    val_params = {'dataloader': dataloaders[1], 
                  'model': model,
                  'loss_fn': loss_fn}
    
    for ep in range(epoch):
        print(f"EPOCH {ep} --------------------------------------------")
        train_loss, train_acc = train_loop(**train_params)
        log['train_loss'].append(train_loss)
        log['train_acc'].append(train_acc)

        val_loss, val_acc = val_loop(**val_params)
        log['val_loss'].append(val_loss)
        log['val_acc'].append(val_acc)
        print(model, optimizer, model_name)
        saving_model(model, optimizer, model_name)
    
    export_figure(log)
    return log