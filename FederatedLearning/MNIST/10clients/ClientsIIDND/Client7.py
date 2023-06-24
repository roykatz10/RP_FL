import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


# AUxillary methods
def getDist(y):
    ax = sns.countplot(y)
    ax.set(title="Count of data classes")
    plt.show()

def getData(from_dist, to_dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<to_dist[y[i]]:
            if counts[y[i]]>from_dist[y[i]]:
                dx.append(x[i]) 
                dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)

# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
K.set_value(model.optimizer.learning_rate, 0.00001)

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
from_dist = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
to_dist   = [3800, 3800, 3800, 3800, 3800, 3800, 3800, 3800, 3800, 3800]
x_train, y_train = getData(from_dist, to_dist, x_train, y_train)
getDist(y_train)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    accuracies = np.zeros(500)
    index = 0
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)


        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        self.accuracies[self.index] = accuracy
        self.index += 1
        if(self.index == 500):
            plt.plot(self.accuracies)
            plt.xlabel("Epoch number")
            plt.ylabel("Accuracy")
            plt.title("Accuracy of FedProx CNN on IID dataset non-equal distribution")
            plt.show()
            print(self.accuracies)



        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)