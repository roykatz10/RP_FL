import flwr as fl
import torch
import argparse
import numpy as np
from torch.utils.data import TensorDataset
import os
import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))
from typing import Dict, List, Optional, Tuple
from src.ADMM_client import ADMM_Net as Net
from src.ADMM_strategy import ADMMStrategy
import matplotlib.pyplot as plt
from src.utils import get_test_data
import time
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--nro", type=int, default = 10)
parser.add_argument("--strat", type=int, default = 0)
parser.add_argument("--nc", type=int, default=2)
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--fn", type=str, default="output.txt")
parser.add_argument("--dset", type=str, default="MNIST_10c")
parser.add_argument("--central", type=int, default=0)
args = parser.parse_args()

# typecast bc argparse is a piece of shit
args.central = bool(args.central)

print(f'args.central: {args.central}')
if args.central==True:
    NUM_CLIENTS = 1
else: 
    NUM_CLIENTS = args.nc
NUM_ROUNDS = args.nro
DEVICE = torch.device("cpu")  # use cpu for all evaluation purposes

server_accuracies = np.zeros(NUM_ROUNDS)

X_test, y_test = get_test_data(args.dset, device= DEVICE)
# X_test = torch.load(f"{dir_path}/../MNIST/10clients/Data/x_test.pt")
# y_test = torch.load(f"{dir_path}/../MNIST/10clients/Data/y_test.pt")
testloader = TensorDataset(X_test, y_test)
params  = Net(0.1, 0.5, args.dset).get_parameters()


# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:\

    net = Net(0.1, 0.5, dset=args.dset).to(DEVICE)
    net = net.double()
    net.set_parameters(parameters)  # Update model with the latest parameters
    loss, accuracy = net.test(testloader)
    server_accuracies[server_round - 1] = accuracy
    print('round:     ', server_round)
    if(server_round == NUM_ROUNDS):
        plt.plot(server_accuracies)
        plt.xlabel("Epoch number")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of FedProx CNN on IID dataset equal distribution")
        # plt.show()
        print('server_accuracies: ', server_accuracies)
        with open(dir_path + "/results/" + args.fn.replace("txt","npy"), "wb") as f:
            np.save(f, server_accuracies)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


# Create FedAvg strategy
strategy_avg = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_fn=evaluate,
)

strategy_prox = fl.server.strategy.FedProx(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    proximal_mu = 0.0005,
    evaluate_fn = evaluate,
)

strategy_median = fl.server.strategy.FedMedian(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)

strategy_yogi = fl.server.strategy.FedYogi(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    initial_parameters=fl.common.ndarrays_to_parameters(params),
    evaluate_fn = evaluate,
)

strategy_qFedAvg = fl.server.strategy.QFedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)

strategy_ADMM = ADMMStrategy(
    rho = args.rho,
    fraction_fit=1,  # Sample 100% of available clients for training
    fraction_evaluate=0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all clients are available
    evaluate_fn = evaluate,
)



if args.strat == 0:
    print('server strategy: fedAvg')
    strat = strategy_avg
elif args.strat == 1:
    print('server strategy: fedMedian')
    strat = strategy_median
elif args.strat == 2:
    print('server strategy: fedProx')
    strat = strategy_prox
elif args.strat == 3:
    print('server strategy: qFedAvg')
    strat = strategy_qFedAvg
elif args.strat == 4:
    print('server strategy: fedYogi')
    strat = strategy_yogi
elif args.strat == 5:
    print('server strategy: ADMM')
    strat = strategy_ADMM

print(f'number of clients: {NUM_CLIENTS}')
if __name__ == "__main__":

    #stdout_orig = sys.stdout
    #sys.stdout = open("output.txt", "w")
    logger = logging.getLogger("flwr")
    logger.addHandler(logging.FileHandler(args.fn))
    now = time.time()
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), strategy=strat)
    #sys.stdout.close()
    #sys.stdout = stdout_orig
    print(f'took {time.time() - now} seconds for {NUM_CLIENTS} clients and {args.nro} rounds')
