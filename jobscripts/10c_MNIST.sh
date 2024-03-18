#!/bin/sh


# hyperparameters (loop over (some of) these later on)
nc=10
nro=400
str=0
lr=0.0001
rho=0.5
dset="MNIST_10c"
iid=0
ed=0
nruns=10



for i in $(seq 1 $nruns);
do
  sbatch 10c_MNIST.sbatch -s $i
done


