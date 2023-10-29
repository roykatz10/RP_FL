import flwr as fl
import torch
import argparse
import numpy as np
from torch.utils.data import TensorDataset
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))
from typing import Dict, List, Optional, Tuple
from multipleiid2 import Net
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--nro", type=int, default = 500)
parser.add_argument("--strat", type=int, default = 0)
args = parser.parse_args()

NUM_ROUNDS = args.nro
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

server_accuracies = np.zeros(NUM_ROUNDS)
X_test = torch.load(f"{dir_path}/Data/x_test.pt")
y_test = torch.load(f"{dir_path}/Data/y_test.pt")
testloader = TensorDataset(X_test, y_test)
params  = Net(0.1, use_admm=False).get_parameters()


# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:\

    net = Net(0.1).to(DEVICE)
    net.set_parameters(parameters)  # Update model with the latest parameters
    loss, accuracy = net.test(testloader)
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


if args.strat == 0:
    strat = strategy_avg
elif args.strat == 1:
    strat = strategy_median
elif args.strat == 2:
    strat = strategy_prox
elif args.strat == 3:
    strat = strategy_qFedAvg
elif args.strat == 4:
    strat = strategy_yogi


if __name__ == "__main__":
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), strategy=strat)