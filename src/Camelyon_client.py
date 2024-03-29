import flwr as fl
from torchvision.models import resnet18
from torch.utils.data import TensorDataset

import numpy as np
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple
from collections import OrderedDict






if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEVICE = torch.device(device)  # Try "  cuda" to train on GPU


class Camelyon_client(fl.client.NumPyClient):

    def __init__(self, X_train, y_train, lr, str, rho):
        self.net = resnet18(pretrained=False)
        self.net.fc = nn.Linear(512, 2)
        self.net = self.net.double()
        if str == 5: # initialize the extra variable for ADMM if needed
            self.rho = rho
            self.net.y = OrderedDict()
            for para, param in zip(self.net.parameters(), self.net.state_dict()):
                self.net.y[param] = torch.zeros(para.shape)


        train_loader = TensorDataset(X_train, y_train)
        self.trainloader = train_loader
        self.lr = lr
        self.str = str

    def train(self, opt, epochs=1):
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                #print("---------------")
                #print(f'image shape: {images.shape}')
                #print("---------------")
                images = images.view(1, 3, 96, 96) 
                outputs = self.net.forward(images)
                labels = labels.long()
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(self.trainloader)
            epoch_acc = correct / total
    
    def test(self, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        # net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images = images.double()
                images = images.view(1, 3, 96, 96)
                labels = labels.long()
                #print(f'image shape: {images.shape}')
                outputs = self.net.forward(images)
                # labels = labels.view(1, -1)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader)
        accuracy = correct / total
        return loss, accuracy
    
    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)

        if str == 5:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k, v in params_dict})
            self.train_admm(config, state_dict, opt)
            params_tobytes = OrderedDict()
            for key in self.net.y.keys():
            #print(f'fit: y param {key} type : {type(self.net.y[key])}')
                params_tobytes[key] = self.net.y[key].detach().numpy().tolist()

            return_dict = {"Y" : params_tobytes}
        else:
            self.train(opt, epochs=1)
            return_dict = {}
        return self.get_parameters({}), len(self.trainloader), return_dict           

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray], config):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        
        state_dict = OrderedDict({k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
        for k, v in params_dict})
        #print(f'state_dict length: {len(state_dict)}')
        self.net.load_state_dict(state_dict, strict = True)


    def train_admm(self, config, z, opt, epochs = 1):
        #self.local_model = np.copy(parameters_to_ndarrays(self.parameters))
        self.update_y(z = z)
        # self.set_parameters(z, {})
        for e in range(epochs):
            for images, labels in self.trainloader:
                opt.zero_grad()
                out = self.net.forward(images)
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

        for para, param in zip(self.net.parameters(), self.net.state_dict()):
            z_t = z[param]
            sub = torch.reshape(torch.sub(para, z_t), (-1,))
            y_res = torch.reshape(self.net.y[param], (-1,))
            fo += torch.dot(y_res.double(), sub.double())
            so += torch.dot(sub, sub)
        so = so * (self.rho/2)

        return (fx_loss + fo + so)
    
    def update_y(self, z):
        for para, param in zip(self.net.parameters(), self.net.y.keys()):
            para_np = para.detach().numpy()
            self.net.y[param] = np.add(self.net.y[param], (self.rho * np.subtract(para_np, z[param])))




