import numpy as np
from src.utils import get_test_data
from src.ADMM_client import ADMM_Net as  Net
from torch.utils.data import TensorDataset

dset= "MNIST_10c"
X_test, y_test = get_test_data(dset)


net = Net(0.1, 0.5, dset=dset)
net = net.double()

testloader = TensorDataset(X_test, y_test)

loss, accuracy = net.test(testloader)
print(loss, accuracy)