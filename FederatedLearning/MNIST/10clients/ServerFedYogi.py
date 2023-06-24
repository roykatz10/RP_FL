import flwr as fl
import sys
import numpy as np
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

strategy = fl.server.strategy.FedYogi(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=10,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    initial_parameters= fl.common.ndarrays_to_parameters(model.get_weights())
)


# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=500) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)