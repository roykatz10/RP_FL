import tensorflow as tf
import numpy as np
import torch
(x_train, y_train_loader), (x_test, y_test_loader) = tf.keras.datasets.mnist.load_data()


x_train = x_train.reshape((-1, 1, 28, 28)).astype(np.float32)
x_test = x_test.reshape((-1, 1, 28, 28)).astype(np.float32)
y_train_loader = y_train_loader.astype(np.int64)
y_test_loader = y_test_loader.astype(np.int64)
x_train_loader = x_train / 255.
x_test = x_test / 255.


torch.save(torch.tensor(x_test), "x_test_full.pt")
torch.save(torch.tensor(y_test_loader), "y_test_full.pt")
torch.save(torch.tensor(x_train), "x_train_full.pt")
torch.save(torch.tensor(y_train_loader), "y_train_full.pt")