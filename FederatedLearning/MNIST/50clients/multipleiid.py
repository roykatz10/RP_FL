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



DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
NUM_ROUNDS = 500
BATCH_SIZE = 1
NUM_CLIENTS = 1




def getData(NUM_CLIENTS, x, y, x_test, y_test):
    dx = [[] for i in range(NUM_CLIENTS)]
    dy = [[] for i in range(NUM_CLIENTS)]

    
    dx_test = []
    dy_test = []
    counts = [0 for i in range(10)]

    for i in range(len(x)):    
        dx[counts[int(y[i])]].append(x[i]) 
        dy[counts[int(y[i])]].append([y[i]])
        counts[int(y[i])] += 1
        counts[int(y[i])] = counts[int(y[i])] % NUM_CLIENTS
    
    for i in range(len(x_test)):
        dx_test.append(x_test[i])
        dy_test.append([y_test[i]])


    datasets = []
    
    for i in range (NUM_CLIENTS):
        tensor_x = torch.Tensor(dx[i]) # transform to torch tensor
        tensor_y = torch.Tensor(dy[i])
        tensor_y = tensor_y.type(torch.LongTensor)

        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset

        datasets.append(my_dataset)

    tensor_x_test = torch.Tensor(dx_test) # transform to torch tensor
    tensor_y_test = torch.Tensor(dy_test)
    tensor_y_test = tensor_y_test.type(torch.LongTensor)

    testloader = TensorDataset(tensor_x_test,tensor_y_test) # create your datset

        
    return datasets, testloader



(x_train, y_train_loader), (x_test, y_test_loader) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 1, 28, 28)).astype(np.float32)
x_test = x_test.reshape((-1, 1, 28, 28)).astype(np.float32)
y_train_loader = y_train_loader.astype(np.int64)
y_test_loader = y_test_loader.astype(np.int64)
x_train_loader = x_train / 255.
x_test = x_test / 255.
# x_train_loader, x_test_loader = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

trainloaders, testloader = getData(NUM_CLIENTS, x_train_loader, y_train_loader, x_test, y_test_loader)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00002)

    # optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            # print(images)
            outputs = net(images)
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



def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = trainloader

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

server_accuracies = np.zeros(NUM_ROUNDS)

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    server_accuracies[server_round - 1] = accuracy
    print('round:     ', server_round)
    if(server_round == NUM_ROUNDS):
        plt.plot(server_accuracies)
        plt.xlabel("Epoch number")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of FedProx CNN on IID dataset equal distribution")
        plt.show()
        print('server accuracies: ', server_accuracies)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


# Create FedAvg strategy
strategy_avg = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=1,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=1,  # Wait until all 10 clients are available
    evaluate_fn=evaluate,
)

strategy_prox = fl.server.strategy.FedProx(
    fraction_fit=0.13,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=7,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=7,  # Wait until all 10 clients are available
    proximal_mu = 0.0005,
    evaluate_fn = evaluate,
)

strategy_median = fl.server.strategy.FedMedian(
    fraction_fit=0.13,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=7,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=7,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)
params = get_parameters(Net())

strategy_yogi = fl.server.strategy.FedYogi(
    fraction_fit=0.13,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=7,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=7,  # Wait until all 10 clients are available
    initial_parameters=fl.common.ndarrays_to_parameters(params),
    evaluate_fn = evaluate,
)

strategy_qFedAvg = fl.server.strategy.QFedAvg(
    fraction_fit=0.13,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=7,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=7,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)


# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy_avg,
    client_resources=client_resources,
)