import flwr as fl
from client import FlowerClient, Net
import argparse
import os
import sys
from torch.utils.data import TensorDataset
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--nro", type=int, default = 5)
# parser.add_argument("--nc", type=int, default = 50)

# 0 = FedAvg
# 1 = FedProx
# 2 = FedMedian
# 3 = FedYogi
# 4 = qFedAvg
parser.add_argument("--strat", type=int, default = 0)

# 0 = 50 clients iid
# 1 = 50 clients niid
# 2 = 10 clients iid ed
# 3 = 10 clients niid ed
# 4 = 10 clients iid ned
# 5 = 10 clients niid ned
parser.add_argument("--scen", type=int, default = 0)
args = parser.parse_args()

NUM_ROUNDS = args.nro
# NUM_CLIENTS = args.nc
NUM_STRAT = args.strat
NUM_SCENARIO = args.scen
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

if NUM_SCENARIO == 0:
    scen_dir = "../../Data/Data_MNIST/Data_50clients_iid"
    NUM_CLIENTS = 50
elif NUM_SCENARIO == 1:
    scen_dir = "../../Data/Data_MNIST/Data_50clients_niid"
    NUM_CLIENTS = 50
elif NUM_SCENARIO == 2:
    scen_dir = "../../Data/Data_MNIST/Data_10clients_iid_ed"
    NUM_CLIENTS = 10
elif NUM_SCENARIO == 3:
    scen_dir = "../../Data/Data_MNIST/Data_10clients_niid_ed"
    NUM_CLIENTS = 10
elif NUM_SCENARIO == 4:
    scen_dir = "../../Data_10clients_iid_ned"
    NUM_CLIENTS = 10
elif NUM_SCENARIO == 5:
    scen_dir = "../../Data/Data_MNIST/Data_10clients_niid_ned"
    NUM_CLIENTS = 10



server_accuracies = np.zeros(NUM_ROUNDS)
# print(f"{dir_path}/Data/x_test.pt")
X_test = torch.load(f"{dir_path}/{scen_dir}/x_test.pt")
y_test = torch.load(f"{dir_path}/{scen_dir}/y_test.pt")
testloader = TensorDataset(X_test, y_test)
params  = Net().get_parameters()


def evaluate(server_round: int,
             parameters: fl.common.NDArrays,
             config: Dict[str, fl.common.Scalar],
             ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net().to(DEVICE)
    net.set_parameters(parameters)  # Update model with the latest parameters
    loss, accuracy = net.test(testloader)
    server_accuracies[server_round - 1] = accuracy
    print('round:     ', server_round)
    if (server_round == NUM_ROUNDS):
        plt.plot(server_accuracies)
        plt.xlabel("Epoch number")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of FedProx CNN on IID dataset equal distribution")
        plt.show()
        print('server accuracies: ', server_accuracies)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

if NUM_STRAT == 0:
    strat = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        evaluate_fn=evaluate,
    )
elif NUM_STRAT == 1:
    strat = fl.server.strategy.FedProx(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        proximal_mu=0.0005,
        evaluate_fn=evaluate,
    )
elif NUM_STRAT == 2:
    strat = fl.server.strategy.FedMedian(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        evaluate_fn=evaluate,
    )
elif NUM_STRAT == 3:
    strat = fl.server.strategy.FedYogi(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
    )
elif NUM_STRAT == 4:
    strat = fl.server.strategy.QFedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        evaluate_fn=evaluate,
    )




def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    X_train = torch.load(f"{dir_path}/{scen_dir}/X_train_id{cid}.pt")
    y_train = torch.load(f"{dir_path}/{scen_dir}/y_train_id{cid}.pt")
    return FlowerClient(X_train, y_train)


# server = server(testloader, NUM_ROUNDS)
# (strategy_avg, strategy_prox, strategy_median, strategy_yogi,  strategy_qFedAvg) = server.strategies()

client_resources = {"num_cpus": 2}


print(f'starting with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds')
# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strat,
    client_resources=client_resources,
)