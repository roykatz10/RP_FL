import flwr as fl
from src.ADMM_strategy import ADMMStrategy
import torch
import numpy as np
import os
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# sys.path.insert(1, os.path.join(sys.path[0], '../common'))



# parent_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(parent_dir)

# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# from run_server import evaluate
# from fifty_clients import run_sim
# from fifty_clients import multipleiid2
from src.Default_client import Net, FlowerClient
from src.utils import from_file
from torch.utils.data import TensorDataset


# from common.client import Net, FlowerClient
# from common.utils import from_file, evaluate
from Roy.RP_FL.src.ADMM_client import ADMM_FlowerClient


NUM_CLIENTS = 10
NUM_ROUNDS = 10
LEARNING_RATE = 0.0001
RHO = 0
DEVICE = torch.device("cpu")
client_resources = {"num_cpus": 1}
server_accuracies = np.zeros(NUM_ROUNDS)
dir_path = os.path.dirname(os.path.realpath(__file__))




# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    

    data_path = dir_path + "/../MNIST/10clients"
    X_test = torch.load(f"{data_path}/Data/x_test.pt")
    y_test = torch.load(f"{data_path}/Data/y_test.pt")
    testloader = TensorDataset(X_test, y_test)
    # params  = Net(0.1).get_parameters()



    net = Net(0.1).to(DEVICE)
    # print(parameters)
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

strategy = ADMMStrategy(
    rho = RHO,
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=1,  # Wait until all 10 clients are available
    evaluate_fn=evaluate,
)


strategy_avg = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=1,  # Wait until all 10 clients are available
    evaluate_fn=evaluate,
)

def client_fn(cid: str) -> ADMM_FlowerClient:
    X_train, y_train = from_file(cid)
    return ADMM_FlowerClient(X_train, y_train, LEARNING_RATE, RHO)

print(f'starting with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds')
# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    #client_resources=client_resources,
)