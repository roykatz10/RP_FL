from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import MNIST

from typing import Dict, List, Optional, Tuple



import flwr as fl
from flwr.common import Metrics
import tensorflow as tf

from multipleiid import getData


NUM_CLIENTS = 10

(x_train, y_train_loader), (x_test, y_test_loader) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1, 1, 28, 28)).astype(np.float32)
x_test = x_test.reshape((-1, 1, 28, 28)).astype(np.float32)
y_train_loader = y_train_loader.astype(np.int64)
y_test_loader = y_test_loader.astype(np.int64)
x_train_loader = x_train / 255.
x_test = x_test / 255.


X_trains, y_trains, x_test, y_test = getData(NUM_CLIENTS, x_train_loader, y_train_loader, x_test, y_test_loader)

#print(X_train[0].size())
#print(x_test.size())


for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
    torch.save(X_train, f'10c_X_train_id{i}.pt')
    torch.save(y_train, f'10c_y_train_id{i}.pt')

torch.save(x_test, "10c_x_test.pt")
torch.save(y_test, "10c_y_test.pt")
