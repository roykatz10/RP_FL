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
import json
from typing import Dict, List, Optional, Tuple

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

dir_path = os.path.dirname(os.path.realpath(__file__))
from src.Default_client import Net, FlowerClient


import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar
#import tensorflow as tf





if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEVICE = torch.device(device)  # Try "  cuda" to train on GPU

#print(len(trainloaders[0]))
class ADMM_Net(Net):
    def __init__(self, lr, rho, dset):
        super(ADMM_Net, self).__init__(dset=dset)
        # init y
        self.rho = rho
        self.y = OrderedDict()      
        for para, param in zip(self.parameters(), self.state_dict()):
            self.y[param] = torch.zeros(para.shape)
            #self.y = np.copy(parameters_to_ndarrays(self.parameters()))

    def train_admm(self, trainloader, z, opt, epochs = 1):

        for e in range(epochs):
            for images, labels in trainloader:
                opt.zero_grad()
                out = self.forward(images)
                labels = labels.long()
                loss = self.admm_loss(out, labels, z)
                # lf = nn.CrossEntropyLoss()
                # loss = lf(out, labels)
                loss.backward()
                opt.step()
        return loss
        

    def admm_loss(self, out, label, z):
        fx = nn.CrossEntropyLoss()
        fx_loss = fx(out, label)

        fo = 0
        so = 0

        for para, param in zip(self.parameters(), self.state_dict()):
            z_t = z[param]
            sub = torch.reshape(torch.sub(para, z_t), (-1,))
            y_res = torch.reshape(self.y[param], (-1,))
            fo += torch.dot(y_res.double(), sub.double())
            so += torch.dot(sub, sub)
        so = so * (self.rho/2)

        return (fx_loss + fo + so)
    
    def update_y(self, z):
        for para, param in zip(self.parameters(), self.y.keys()):
            para_np = para.detach().numpy()
            self.y[param] = np.add(self.y[param], (self.rho * np.subtract(para_np, z[param])))



class ADMM_FlowerClient(FlowerClient):
    def __init__(self, X_train, y_train, lr, rho, dset):
        super().__init__(X_train, y_train, lr, dset)
        self.net = ADMM_Net(lr, rho, dset=dset).to(device=DEVICE)
        self.net = self.net.double()
        self.lr = lr

    # this does not give y at the moment, could consider changing that
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.net.get_parameters()

    #admm uses 'local' params
    # def set_parameters(self, parameters, config):
        # pass

    def fit(self, parameters, config
    ) -> Tuple[NDArrays, int, Dict[str, OrderedDict]]:
        self.set_parameters(parameters, config)
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.train_admm(config, state_dict, opt)
        # self.train_admm(config, parameters, opt)
        params = self.get_parameters({})
        params_tobytes = OrderedDict()
        for key in self.net.y.keys():
            #print(f'fit: y param {key} type : {type(self.net.y[key])}')
            params_tobytes[key] = self.net.y[key].detach().numpy().tolist()

        y_bytes = json.dumps(params_tobytes).encode("utf-8")
        #return params, len(self.trainloader), {"Y" : y_bytes}
        return params, len(self.trainloader), {"Y" : y_bytes}
    
    def train_admm(self, config, z, opt, until_convergence = False, t = 0, max_epochs = 100):
        #self.local_model = np.copy(parameters_to_ndarrays(self.parameters))
        self.net.update_y(z = z)
        # self.set_parameters(z, {})
        error = 10
        lepochs = 0
        # print(config)
        # print(f'until convergence: {until_convergence}')
        if until_convergence == False:
            max_epochs = 1
        while (error > t) and (lepochs < max_epochs):
            # print(f'local epoch: {lepochs}')
            self.net.train_admm(self.trainloader, z, opt, epochs = 1)
            error, _ = self.net.test(self.valloader)
            lepochs += 1
