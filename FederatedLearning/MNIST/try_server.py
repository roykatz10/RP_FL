# import flwr as fl
# from client import FlowerClient, Net, from_file
# # from multipleiid import get_parameters
# import torch
# from typing import Dict, List, Optional, Tuple
# import numpy as np
# import os
# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# dir_path = os.path.dirname(os.path.realpath(__file__))
# import matplotlib.pyplot as plt
#
#
#
#
#
#
#
# class server():
#     def __init__(self, testLoader, NUM_ROUNDS):
#
#         self.server_accuracies = np.zeros(NUM_ROUNDS)
#         self.DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
#         self.testloader = testLoader
#         self.params  = Net().get_parameters()
#
#     def evaluate(self,
#         server_round: int,
#         parameters: fl.common.NDArrays,
#         config: Dict[str, fl.common.Scalar],
#     ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#
#         net = Net().to(self.DEVICE)
#         net.set_parameters(parameters)  # Update model with the latest parameters
#         loss, accuracy = net.test(self.testloader)
#         self.server_accuracies[server_round - 1] = accuracy
#         print('round:     ', server_round)
#         if(server_round == self.NUM_ROUNDS):
#             plt.plot(self.server_accuracies)
#             plt.xlabel("Epoch number")
#             plt.ylabel("Accuracy")
#             plt.title("Accuracy of FedProx CNN on IID dataset equal distribution")
#             plt.show()
#             print('server accuracies: ', self.server_accuracies)
#         print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
#         return loss, {"accuracy": accuracy}
#
#     # Create FedAvg strategy
#     def strategies(self):
#         strategy_avg = fl.server.strategy.FedAvg(
#             fraction_fit=1.0,  # Sample 100% of available clients for training
#             fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
#             min_fit_clients=1,  # Never sample less than 10 clients for training
#             min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
#             min_available_clients=1,  # Wait until all 10 clients are available
#             evaluate_fn= self.evaluate,
#         )
#         strategy_prox = fl.server.strategy.FedProx(
#             fraction_fit=1.0,  # Sample 100% of available clients for training
#             fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
#             min_fit_clients=1,  # Never sample less than 10 clients for training
#             min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
#             min_available_clients=1,  # Wait until all 10 clients are available
#             proximal_mu = 0.0005,
#             evaluate_fn = self.evaluate,
#         )
#         strategy_median = fl.server.strategy.FedMedian(
#             fraction_fit=1.0,  # Sample 100% of available clients for training
#             fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
#             min_fit_clients=1,  # Never sample less than 10 clients for training
#             min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
#             min_available_clients=1,  # Wait until all 10 clients are available
#             evaluate_fn = self.evaluate,
#         )
#         strategy_yogi = fl.server.strategy.FedYogi(
#             fraction_fit=0.13,  # Sample 100% of available clients for training
#             fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
#             min_fit_clients=7,  # Never sample less than 10 clients for training
#             min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
#             min_available_clients=7,  # Wait until all 10 clients are available
#             initial_parameters=fl.common.ndarrays_to_parameters(self.params),
#             evaluate_fn=self.evaluate,
#         )
#
#         strategy_qFedAvg = fl.server.strategy.QFedAvg(
#             fraction_fit=1.0,  # Sample 100% of available clients for training
#             fraction_evaluate=0.0,  # Sample 50% of available clients for evaluation
#             min_fit_clients=1,  # Never sample less than 10 clients for training
#             min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
#             min_available_clients=1,  # Wait until all 10 clients are available
#             evaluate_fn = self.evaluate,
#         )
#
#         return (strategy_avg, strategy_prox, strategy_median, strategy_yogi,  strategy_qFedAvg)
#
#
#
