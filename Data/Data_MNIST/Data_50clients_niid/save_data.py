
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, random_split, TensorDataset



def getData(NUM_CLIENTS, x, y, x_test, y_test):
    dx = [[] for i in range(NUM_CLIENTS)]
    dy = [[] for i in range(NUM_CLIENTS)]
    dx_test = []
    dy_test = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        location = counts[int(y[i])] + int(NUM_CLIENTS / 5) * (int(y[i] / 2))
        # if len(dx[location]) < 1000:
        dx[location].append(x[i])

        dy[location].append([y[i]])

        counts[int(y[i])] += 1
        counts[int(y[i])] = counts[int(y[i])] % int((NUM_CLIENTS / 5))

    for i in range(len(x_test)):
        dx_test.append(x_test[i])
        dy_test.append([y_test[i]])

    X_train = []
    y_train = []
    for i in range(NUM_CLIENTS):
        tensor_x = torch.Tensor(dx[i])  # transform to torch tensor
        tensor_y = torch.Tensor(dy[i])
        tensor_y = tensor_y.type(torch.LongTensor)

        # my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset

        X_train.append(tensor_x)
        y_train.append(tensor_y)

    tensor_x_test = torch.Tensor(dx_test)  # transform to torch tensor
    tensor_y_test = torch.Tensor(dy_test)
    tensor_y_test = tensor_y_test.type(torch.LongTensor)

    # testloader = TensorDataset(tensor_x_test,tensor_y_test) # create your datset

    return X_train, y_train, tensor_x_test, tensor_y_test


NUM_CLIENTS = 50

(x_train, y_train_loader), (x_test, y_test_loader) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1, 1, 28, 28)).astype(np.float32)
x_test = x_test.reshape((-1, 1, 28, 28)).astype(np.float32)
y_train_loader = y_train_loader.astype(np.int64)
y_test_loader = y_test_loader.astype(np.int64)
x_train_loader = x_train / 255.
x_test = x_test / 255.


X_trains, y_trains, x_test, y_test = getData(NUM_CLIENTS, x_train_loader, y_train_loader, x_test, y_test_loader)

#print(X_train[0].size())
#print(x_test.size())


for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
    torch.save(X_train, f'X_train_id{i}.pt')
    torch.save(y_train, f'y_train_id{i}.pt')

torch.save(x_test, "x_test.pt")
torch.save(y_test, "y_test.pt")
