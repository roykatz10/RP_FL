import torch
import numpy as np
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

## Mapping:
# ChemDB - 0
# PKIS   - 1
# Tang   - 2


dsets = ["ChemDB_FLT3_processed.csv", "PKIS_FLT3_processed.csv", "Tang_FLT3_processed.csv"]


for dset_id, dset in enumerate(dsets):
    df = pd.read_csv(dir_path + "/" +  dset)
    train_df = df.loc[df['test/train'] == "train"] 
    X_train = train_df.drop(columns = ['label', "test/train"]).values
    y_train = train_df['label'].values

    test_df = df.loc[df['test/train'] == "test"]
    X_test = test_df.drop(columns = ['label', "test/train"]).values
    y_test = test_df['label'].values

    if dset_id == 0:
        X_test_full = X_test.copy()
        y_test_full = y_test.copy()
    else:
        X_test_full = np.concatenate((X_test_full, X_test))
        y_test_full = np.concatenate((y_test_full, y_test))

    torch.save(torch.tensor(X_train), dir_path + f'/X_train_id{dset_id}.pt')
    torch.save(torch.tensor(y_train), dir_path + f'/y_train_id{dset_id}.pt')
    torch.save(torch.tensor(X_test), dir_path + f'/X_test_id{dset_id}.pt')
    torch.save(torch.tensor(y_test), dir_path + f'/y_test_id{dset_id}.pt')

torch.save(torch.tensor(X_test_full), dir_path + f'/x_test.pt')
torch.save(torch.tensor(y_test_full), dir_path + f'/y_test.pt')
