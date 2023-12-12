import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
import os
from typing import Dict, List, Optional, Tuple
import torch
import argparse



sys.path.insert(1, os.path.join(sys.path[0], '../..'))
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--nro", type=int, default = 5)

# 0 = FedAvg
# 1 = FedProx
# 2 = FedMedian
# 3 = FedYogi
# 4 = qFedAvg
parser.add_argument("--strat", type=int, default = 0)


args = parser.parse_args()

NUM_ROUNDS = args.nro
NUM_STRAT = args.strat

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 3


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8192, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
K.set_value(model.optimizer.learning_rate, 0.00001)

data_1 = pd.read_csv('../../Data/Data_kinase/ChemDB_FLT3_processed.csv')
data_2 = pd.read_csv('../../Data/Data_kinase/PKIS_FLT3_processed.csv')
data_3 = pd.read_csv('../../Data/Data_kinase/Tang_FLT3_processed.csv')

all_data = pd.concat([data_1, data_2, data_3])



x_test = all_data.drop('label', axis=1).loc[all_data['test/train'] == 'test'].drop('test/train', axis=1).values
y_test = all_data.loc[all_data['test/train'] == 'test']["label"].values



accuracies0 = []
accuracies1 = []
accuracies2 = []

class FlowerClient(fl.client.NumPyClient):


    def __init__(self, x_train,  y_train, x_test, y_test, model, partition_id):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.partition_id = partition_id


    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        r = self.model.fit(self.x_train, self.y_train, epochs=1, validation_data=(self.x_test, self.y_test), verbose=0, batch_size=2)
        hist = r.history
        print("Fit history : " ,hist)

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Eval accuracy : ", accuracy)

        print('id: ', self.partition_id, int(self.partition_id))
        if(int(self.partition_id) == 0):
            print('wrote to file')
            with open('readme0.txt', 'a') as f:
                f.write(str(accuracy))
                f.write(',')
        if(int(self.partition_id) == 1):
            print('wrote to file')
            with open('readme1.txt', 'a') as f:
                f.write(str(accuracy))
                f.write(',')
        if(int(self.partition_id) == 2):
            with open('readme2.txt', 'a') as f:
                f.write(str(accuracy))
                f.write(',')
        return loss, len(self.x_test), {"accuracy": accuracy}




def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8192, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    K.set_value(model.optimizer.learning_rate, 0.00001) 

    x_train = np.array([])
    y_train = np.array([])
    x_validation = np.array([])
    y_validation = np.array([])

    
    # print('cid: x ', cid)
    if cid == '0': 
        x_train = data_1.drop('label', axis=1).loc[data_1['test/train'] == 'train'].drop('test/train', axis=1).values
        y_train = data_1.loc[data_1['test/train'] == 'train']["label"].values
        x_validation = data_1.drop('label', axis=1).loc[data_1['test/train'] == 'test'].drop('test/train', axis=1).values
        y_validation = data_1.loc[data_1['test/train'] == 'test']["label"].values
    if cid == '1': 
        x_train = data_2.drop('label', axis=1).loc[data_2['test/train'] == 'train'].drop('test/train', axis=1).values
        y_train = data_2.loc[data_2['test/train'] == 'train']["label"].values
        x_validation = data_2.drop('label', axis=1).loc[data_2['test/train'] == 'test'].drop('test/train', axis=1).values
        y_validation = data_2.loc[data_2['test/train'] == 'test']["label"].values

    if cid == '2': 
        x_train = data_3.drop('label', axis=1).loc[data_3['test/train'] == 'train'].drop('test/train', axis=1).values
        y_train = data_3.loc[data_3['test/train'] == 'train']["label"].values
        x_validation = data_3.drop('label', axis=1).loc[data_3['test/train'] == 'test'].drop('test/train', axis=1).values
        y_validation = data_3.loc[data_3['test/train'] == 'test']["label"].values

    # print('x_train  : ', x_train)


    # Create a  single Flower client representing a single organization
    return FlowerClient(x_train, y_train, x_validation, y_validation, model, cid)

server_accuracies = np.zeros(NUM_ROUNDS)

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(8192, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    K.set_value(model.optimizer.learning_rate, 0.00001)
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
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


strategy_fedavg = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 10 clients for training
    min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
    min_available_clients=3,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)


strategy_fedprox = fl.server.strategy.FedProx(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 10 clients for training
    min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
    min_available_clients=3,  # Wait until all 10 clients are available
    proximal_mu = 0.0005,
    evaluate_fn = evaluate,
)


strategy_fedmedian = fl.server.strategy.FedMedian(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 10 clients for training
    min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
    min_available_clients=3,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)

strategy_fedyogi = fl.server.strategy.FedYogi(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=3,  # Never sample less than 10 clients for training
    min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
    min_available_clients=3,  # Wait until all 10 clients are available
    initial_parameters= fl.common.ndarrays_to_parameters(model.get_weights()),
    evaluate_fn = evaluate,
)

strategy_qfedavg = fl.server.strategy.QFedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
        min_fit_clients=3,  # Never sample less than 10 clients for training
        min_evaluate_clients=3,  # Never sample less than 5 clients for evaluation
        min_available_clients=3,  # Wait until all 10 clients are available
        evaluate_fn=evaluate,
    )


if NUM_STRAT == 0:
    strat = strategy_fedavg
elif NUM_STRAT == 1:
    strat = strategy_fedprox
elif NUM_STRAT == 2:
    strat = strategy_fedmedian
elif NUM_STRAT == 3:
    strat = strategy_fedyogi
elif NUM_STRAT == 4:
    strat = strategy_qfedavg


# Start Flower server for three rounds of federated learning
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}




# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=6000.0),
    strategy=strat,
    client_resources=client_resources,
)