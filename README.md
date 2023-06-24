# RP_FL
Aggregation-algorithm-fl-comparison

This project requires flower and can be installed by:
!pip install -q flwr[simulation] torch torchvision matplotlib


The kinase inhibition dataset can be run by running Kinase\simulation.py


The 10 clients MNIST can be run by running the required Server in MNIST\10Clients as well as running the 10 respective clients. 

The 50 clients MNIST can be run with:
cd 50clients
python multipleiid.py

or
cd 50clients
python multipleniid.py
