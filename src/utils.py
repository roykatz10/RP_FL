import torch
import flwr as fl
import torch
import numpy as np
import os
from torch.utils.data import TensorDataset
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

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
        raise(NotImplementedError(f"{__name__}: still need to implement kinase non-sim"))
    elif dset == "camelyon":
        raise(NotImplementedError(f"{__name__}: still need to implement camelyon dataset"))
    else:
        raise(ValueError(f"{__name__}: unknown dataset: {dset}"))
    X_train = torch.load(f"{full_path}/X_train_id{id}.pt", map_location=device) 
    y_train = torch.load(f"{full_path}/y_train_id{id}.pt", map_location=device)
    return X_train, y_train


def get_test_data(dset, device = "cpu"):
    prefix = dir_path + "/../Data/"
    if dset == "MNIST_10c" or dset == "MNIST_50c":
        full_path = prefix + "Data_MNIST/"
    elif dset == "kinase":
        raise(NotImplementedError(f"{__name__}: still need to implement kinase"))
    elif dset == "camelyon":
        raise(NotImplementedError(f"{__name__}: still need to implement camelyon"))
    else:
        raise(ValueError(f"{__name__}: unknown dataset: {dset}"))
    X_train = torch.load(f"{full_path}x_test.pt", map_location=device) 
    y_train = torch.load(f"{full_path}y_test.pt", map_location=device)
    return X_train, y_train
