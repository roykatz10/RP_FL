import flwr as fl
from typing import List, Optional, Tuple, Dict, OrderedDict
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
import json

class ADMMStrategy(fl.server.strategy.FedAvg):

    def __init__(self, rho, fraction_fit, fraction_evaluate, min_fit_clients, min_evaluate_clients, min_available_clients, evaluate_fn):
        super().__init__(
                         fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate, min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients, min_available_clients=min_available_clients, evaluate_fn=evaluate_fn)
        self.rho = rho

    def aggregate_fit(self,rnd: int,
     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
     failures: List[BaseException],
     ) -> Optional[float]:
        # y = np.array([(res.metrics["Y"]) for _, res in results], dtype=object)        

        X_list = []
        y_list = []

        for _, res in results:
            # 
            y_local = json.loads(res.metrics["Y"].decode("utf-8"))
            keys = y_local.keys()
            parameters = res.parameters
            params_dict = zip(keys, parameters_to_ndarrays(parameters))
            state_dict = OrderedDict({k: np.array(v) for k, v in params_dict})
            X_list.append(state_dict)
            y_list.append(y_local)
        y = np.array(y_list, dtype=object)
        X = np.array(X_list, dtype=object)
        # print(f'X: {X[0]}')
        z = self.average_ADMM(X, y)
        z_arr = np.array([z[key] for key in z.keys()], dtype = object)

        return ndarrays_to_parameters(z_arr), {}
    
    def average_ADMM(self,X : np.array, y : np.array) -> dict:
        num_clients = len(y)
        z = {}    
        for para in y[0].keys():
            y_arr = np.zeros((num_clients,) + X[0][para].shape)

            if self.rho != 0:
            # idk why but this way it does work
                for i in range(num_clients):
                    # print(type(y[i][para][0][0]))
                    y_arr[i,...] = np.array(y[i][para]) * (1/self.rho)
            par_list = np.array([X_local[para] for X_local in X])
            # y_list = np.array([y[i][para] for i in range(num_clients)])
            # y_list = y_list * (1/self.rho)
            z[para] = np.sum((par_list + y_arr), axis = 0)/num_clients
        return z



