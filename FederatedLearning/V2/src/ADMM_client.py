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
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays

from typing import Dict, List, Optional, Tuple

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))


import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar
#import tensorflow as tf

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEVICE = torch.device(device)  # Try "cuda" to train on GPU

# print(
#     f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
# )
# NUM_ROUNDS = 500
# BATCH_SIZE = 1
# NUM_CLIENTS = 10





#print(len(trainloaders[0]))
class Net(nn.Module):
    def __init__(self, lr):
        super(Net, self).__init__()
        self.lr = lr
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return  logits
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    #def get_parameters(self, config):
    #    return get_parameters(self.net)
    def set_parameters(self, parameters: List[np.ndarray]):
        # print(parameters)
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)


    def train(self, trainloader, epochs: int, verbose=False, admm= False, **kwargs):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        # optimizer = torch.optim.Adam(net.parameters())
        #net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                # print(images)
                outputs = self.forward(images)
                # if(len(outputs) != labels.dim):
                #     print('fuck:', len(outputs), labels.dim)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test(self, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        #net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.forward(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader)
        accuracy = correct / total
        return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, lr):
        self.net = Net(lr)
        self.lr = lr
        train_loader = TensorDataset(X_train, y_train)
        #test_loader = TensorDataset(X_test, y_test)
        self.trainloader = train_loader
        self.valloader = train_loader

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.net.get_parameters()
    
    def set_parameters(self, parameters, config):
        # if self.use_admm == False: #admm uses local params
        self.net.set_parameters(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        if self.use_admm:
            self.train_admm(config, parameters, self.admm_kwargs)
        else:
            self.net.train(self.trainloader, epochs=1)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        loss, accuracy = self.net.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
       



