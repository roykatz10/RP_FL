import flwr as fl
import argparse
import torch


import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from common.utils import from_file
from src.ADMM_model import ADMM_FlowerClient

#sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--rho", type=float, default=0.5)

args = parser.parse_args()

X_train, y_train = from_file(args.cid)
# X_test = torch.load("Data/x_test.pt")
# y_test = torch.load("Data/y_test.pt")

client = ADMM_FlowerClient(X_train, y_train, args.lr, args.rho)

print(f'booting up client {args.cid}')
fl.client.start_numpy_client(server_address = "[::]:8080", client = client)