from multipleiid2 import FlowerClient, from_file
import flwr as fl
import argparse
import torch


import os
import sys
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, default=0)
parser.add_argument("--ua", type=bool, default=False)

args = parser.parse_args()

X_train, y_train = from_file(args.cid)
# X_test = torch.load("Data/x_test.pt")
# y_test = torch.load("Data/y_test.pt")

client = FlowerClient(X_train, y_train, use_admm = args.ua)

fl.client.start_numpy_client(server_address = "[::]:8080", client = client)