import flwr as fl
from typing import List, Optional, Tuple, Dict
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters

class ADMMStrategy(fl.server.strategy.FedAvg):

    def __init__(self, rho, fraction_fit, fraction_evaluate, min_fit_clients, min_evaluate_clients, min_available_clients, evaluate_fn):
        super().__init__(
                         fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn)
        self.rho = rho

    def aggregate_fit(self,rnd: int,
     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
     failures: List[BaseException],
     ) -> Optional[float]:
        X = np.array([parameters_to_ndarrays(res.parameters) for _, res in results], dtype=object)
        y = np.array([parameters_to_ndarrays(res.y) for _, res in results], dtype=object)        
        return ndarrays_to_parameters(self.average_ADMM(X, y))
    
    def average_ADMM(self,X, y):
        num_clients = len(y)

        z = {}
        # print(f'shapes: {X.shape}, {y.shape}')
        return z  



