import flwr as fl
from multipleiid2 import FlowerClient, Net, from_file
import torch
from run_server import evaluate

NUM_CLIENTS = 50
NUM_ROUNDS = 100
DEVICE = torch.device("cpu")

# Create FedAvg strategy
strategy_avg = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=1,  # Never sample less than 10 clients for training
    min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
    min_available_clients=1,  # Wait until all 10 clients are available
    evaluate_fn=evaluate,
)


# # The `evaluate` function will be by Flower called after every round
# def evaluate(
#     server_round: int,
#     parameters: fl.common.NDArrays,
#     config: Dict[str, fl.common.Scalar],
# ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:\

#     net = Net().to(DEVICE)
#     net.set_parameters(parameters)  # Update model with the latest parameters
#     loss, accuracy = net.test(testloader)
#     server_accuracies[server_round - 1] = accuracy
#     print('round:     ', server_round)
#     if(server_round == NUM_ROUNDS):
#         plt.plot(server_accuracies)
#         plt.xlabel("Epoch number")
#         plt.ylabel("Accuracy")
#         plt.title("Accuracy of FedProx CNN on IID dataset equal distribution")
#         plt.show()
#         print('server accuracies: ', server_accuracies)
#     print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
#     return loss, {"accuracy": accuracy}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    #net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    #trainloader = trainloaders[int(cid)]
    X_train, y_train = from_file(cid)
    #valloader = trainloader

    # Create a  single Flower client representing a single organization
    return FlowerClient(X_train, y_train)

client_resources = {"num_cpus": 1}

print(f'starting with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds')
# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy_avg,
    client_resources=client_resources,
)