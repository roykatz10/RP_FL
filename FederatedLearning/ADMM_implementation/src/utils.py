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


# TODO: change this to be non-hardcoded (for other datasets)
data_path = dir_path + "/../../MNIST/10clients"

def from_file(dset, id, device = "cpu"):
    if dset == "MNIST_10c":
        data_path = dir_path + "/../../MNIST/10clients"
    elif dset == "MNIST_50c":
        data_path = dir_path + "/../../MNIST/50clients"
    elif dset == "kinase":
        data_path = dir_path + "/../.."
    elif dset == "camelyon":
        data_path = dir_path + "/../../camelyon"
    else:
        raise(ValueError("unknown dataset"))
    X_train = torch.load(f"{data_path}/Data/X_train_id{id}.pt", map_location=device) 
    y_train = torch.load(f"{data_path}/Data/y_train_id{id}.pt", map_location=device)
    return X_train, y_train
