import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
from src.utils import get_train_data
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEVICE = torch.device(device)  # Try "cuda" to train on GPU

print(f'device: {DEVICE}')


lr = 0.05
epochs = 10
batch_size = 128


accuracies = np.zeros(epochs)
net = resnet18(weights = None)
net.fc = nn.Linear(512,2)
net = net.double()
net = net.to(device = DEVICE)


X_train, y_train = get_train_data("camelyon", 0)
print("data loaded")
train_loader = DataLoader(TensorDataset(X_train, torch.Tensor(y_train.astype(int)), batch_size = batch_size,  shuffle = True)

opt = torch.optim.SGD(net.parameters(), lr = lr)
crit = torch.nn.CrossEntropyLoss()

for e in tqdm(range(epochs)):

    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        #print(f'X_batch shape: {X_batch.shape}')
        #print(f'y_batch shape: {y_batch.shape}')
        y_batch = y_batch.view(-1)
        y_batch = y_batch.int()
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        opt.zero_grad()
        out = net(X_batch)
    
        loss = crit(out, y_batch)
        loss.backward()
        opt.step()
        epoch_loss += loss
    accuracies[e] =epoch_loss

print(f'total loss for {epoch} epochs, lr {lr} and batch size {batch_size}:')
print(accuracies)
