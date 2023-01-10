# %% [code]
# %% [code]
# %% [code]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch

figure_dir = "./figures"
model_dir = "./models"

def plot(log, pair, fig_name):
    epochs = np.arange(len(log[pair[0]]))
    
    fig, ax = plt.subplots()
    ax.plot(epochs,
            log[pair[0]],
            color="red")
    ax.set_xlabel("epoch", fontsize = 14)
    ax.set_ylabel(pair[0],
                color="red",
                fontsize=14)

    ax2=ax.twinx()
    ax2.plot(epochs, log[pair[1]], color="blue")
    ax2.set_ylabel(pair[1],color="blue",fontsize=14)
    plt.show()
    
    if not os.path.isdir(figure_dir):
        os.mkdir(figure_dir)
    
    fig_name = os.path.join(figure_dir, f"{fig_name}.jpeg")
    fig.savefig(fig_name,
                format='jpeg',
                dpi=100,
                bbox_inches='tight')
    
def saving_model(model, optimizer, model_name):
    state = {'optimizer': optimizer.state_dict(), 'model': model.state_dict()}
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    saving_path = os.path.join(model_dir, model_name)
    torch.save(state, saving_path)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)