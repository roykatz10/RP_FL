import numpy as np
import torch


# load in the (uint8) tensors for each center

X_test_c0 = torch.load("X_test_c0.pt")
X_test_c1 = torch.load("X_test_c1.pt")
X_test_c2 = torch.load("X_test_c2.pt")
X_test_c3 = torch.load("X_test_c3.pt")
X_test_c4 = torch.load("X_test_c4.pt")

#y_train_c0 = torch.load("y_train_c0.pt")
#y_train_c1 = torch.load("y_train_c1.pt")
#y_train_c2 = torch.load("y_train_c2.pt")
#y_train_c3 = torch.load("y_train_c3.pt")
#y_train_c4 = torch.load("y_train_c4.pt")



## do the same for test, but we can do that afterwards or smth to preserver memory
#X_test_c0 = torch.load("X_test_c0.pt")
#X_test_c1 = torch.load("X_test_c1.pt")
#X_test_c2 = torch.load("X_test_c2.pt")
#X_test_c3 = torch.load("X_test_c3.pt")
#X_test_c4 = torch.load("X_test_c4.pt")

# make sure that these are uint8, and not preprocessed (i.e. values until 255)

print(f'center 0 type: {X_test_c0.dtype}, values: {torch.max(X_test_c0)}')
print(f'center 1 type: {X_test_c1.dtype}, values: {torch.max(X_test_c1)}')
print(f'center 2 type: {X_test_c2.dtype}, values: {torch.max(X_test_c2)}')
print(f'center 3 type: {X_test_c3.dtype}, values: {torch.max(X_test_c3)}')
print(f'center 4 type: {X_test_c4.dtype}, values: {torch.max(X_test_c4)}')


# convert uint8 to float (size goes x4!)

X_test_c0 = X_test_c0.float()
X_test_c1 = X_test_c1.float()
X_test_c2 = X_test_c2.float()
X_test_c3 = X_test_c3.float()
X_test_c4 = X_test_c4.float()


# normalize: /255 (and minus 128?)

X_test_c0 = X_test_c0/255
X_test_c1 = X_test_c1/255
X_test_c2 = X_test_c2/255
X_test_c3 = X_test_c3/255
X_test_c4 = X_test_c4/255


# save new tensors for each center
torch.save(X_test_c0, "X_test_c0_prep.pt")
torch.save(X_test_c1, "X_test_c1_prep.pt")
torch.save(X_test_c2, "X_test_c2_prep.pt")
torch.save(X_test_c3, "X_test_c3_prep.pt")
torch.save(X_test_c4, "X_test_c4_prep.pt")


# concatenate all datasets into one (this one will take quite some memory..)
X_test_full = torch.cat((X_test_c0, X_test_c1, X_test_c2, X_test_c3, X_test_c4))

# save full dataset
torch.save(X_test_full, "X_test_full_prep.pt")


