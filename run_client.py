import flwr as fl
import argparse
import torch


import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.utils import get_train_data
from src.ADMM_client import ADMM_FlowerClient
from src.Default_client import FlowerClient

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEVICE = torch.device(device)  # Try "cuda" to train on GPU
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--dset", type=str, default="MNIST_10c")
parser.add_argument("--iid", type=bool, default=True)
parser.add_argument("--ed", type=bool, default=True)
parser.add_argument("--str", type=int, default=0)
args = parser.parse_args()


id = args.cid
# id = os.environ['SLURM_PROCID']


X_train, y_train = get_train_data(args.dset, id, iid = args.iid, ed = args.ed, device=DEVICE)
# X_test = torch.load("Data/x_test.pt")
# y_test = torch.load("Data/y_test.pt")

if args.str == 5:
    client = ADMM_FlowerClient(X_train, y_train, args.lr, args.rho, args.dset)
else:
    client = FlowerClient(X_train, y_train, args.lr, args.dset)
print(f'booting up client {id}')
fl.client.start_numpy_client(server_address = "[::]:8080", client = client)
