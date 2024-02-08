import torch
import flwr as fl
import torch
import numpy as np
import os
from torch.utils.data import TensorDataset
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from torchvision.models import resnet18

import torch.nn as nn
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))



def get_train_data(dset, id, iid = True, ed = True, device = "cpu"):
    prefix = dir_path + "/../Data/"
    if iid:
        suf1 = "iid"
    else:
        suf1 = "niid"
    if ed:
        suf2 = "_ed"
    else: 
        suf2 = "_ned"

    if dset == "MNIST_10c":
        folder = "Data_MNIST/Data_10clients_"
        full_path = prefix + folder + suf1 + suf2
    elif dset == "MNIST_50c":
        folder = "Data_MNIST/Data_50clients_"
        full_path = prefix + folder + suf1
    elif dset == "kinase":
        folder = "Data_kinase"
        full_path = prefix + folder
    elif dset == "camelyon":
        folder = "Data_camelyon"
        full_path = prefix + folder
        
    else:
        raise(ValueError(f"{__name__}: unknown dataset: {dset}"))
    X_train = torch.load(f"{full_path}/X_train_id{id}.pt", map_location=device).double() 
    y_train = torch.load(f"{full_path}/y_train_id{id}.pt", map_location=device)
    y_train = y_train.reshape(-1, 1)
    return X_train, y_train


def get_central_train_data(dset, device='gpu'):
    prefix = dir_path + "/../Data/"
    if (dset=="MNIST_10c") or (dset == "MNIST_50c"):
        full_path = prefix + "Data_MNIST/"
    elif dset == "kinase":
        pass
    elif dset == "camelyon":
        full_path = prefix + "Data_camelyon/"
    else:
        raise(ValueError(f'unkown dataset: {dset}'))
    X_train = torch.load(f"{full_path}x_train_full.pt", map_location=device).double()
    y_train = torch.load(f"{full_path}y_train_full.pt", map_location=device).double()
    y_train = y_train.reshape(-1, 1)
    return X_train, y_train

def get_test_data(dset, device = "cpu"):
    prefix = dir_path + "/../Data/"
    if dset == "MNIST_10c" or dset == "MNIST_50c":
        full_path = prefix + "Data_MNIST/"
    elif dset == "kinase":
        full_path = prefix + "Data_kinase/"
    elif dset == "camelyon":
        full_path = prefix + "Data_camelyon/"
    else:
        raise(ValueError(f"{__name__}: unknown dataset: {dset}"))
    X_test = torch.load(f"{full_path}x_test.pt", map_location=device).double()
    y_test = torch.load(f"{full_path}y_test.pt", map_location=device)
    y_test = y_test.reshape(-1, 1)
    # print(f'y_test shape: {y_train.shape}')
    return X_test, y_test


def get_arch(dset):
    if (dset == "MNIST_10c") or (dset == "MNIST_50c"):
        return nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()
        )
    elif dset == "kinase":
        return nn.Sequential(
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax()
        )
    elif dset == "camelyon":
        raise(ValueError("shouldn't use this for camelyon dataset"))
    else:
        raise(ValueError("unknown dataset"))
