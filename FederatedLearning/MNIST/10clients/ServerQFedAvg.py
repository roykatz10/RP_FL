import flwr as fl
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tensorflow import keras
from keras import backend as K

from typing import Dict, List, Optional, Tuple



import flwr as fl
from flwr.common import Metrics
import tensorflow as tf

NUM_ROUNDS = 500


server_accuracies = np.zeros(NUM_ROUNDS)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
K.set_value(model.optimizer.learning_rate, 0.00001)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
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


strategy = fl.server.strategy.QFedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=10,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    evaluate_fn = evaluate,
)


# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
        
)