import numpy as np
import torch




y_train_c0 = torch.load("Y_train_c0.pt")
y_train_c1 = torch.load("Y_train_c1.pt")
y_train_c2 = torch.load("Y_train_c2.pt")
y_train_c3 = torch.load("Y_train_c3.pt")
y_train_c4 = torch.load("Y_train_c4.pt")

y_test_c0 = torch.load("Y_test_c0.pt")
y_test_c1 = torch.load("Y_test_c1.pt")
y_test_c2 = torch.load("Y_test_c2.pt")
y_test_c3 = torch.load("Y_test_c3.pt")
y_test_c4 = torch.load("Y_test_c4.pt")

print(f'type of y_train: {type(y_train_c0)}, shape: {y_train_c0.shape}')

y_train_full = np.concatenate((y_train_c0, y_train_c1, y_train_c2, y_train_c3, y_train_c4))
y_test_full = np.concatenate((y_test_c0, y_test_c1, y_test_c2, y_test_c3, y_test_c4))

torch.save(y_train_full, "Y_train_full.pt")
torch.save(y_test_full, "Y_test_full.pt")


