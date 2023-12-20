# RP_FL
This project requires flower and can be installed with: 

!pip install -q flwr[simulation] torch torchvision matplotlib 
### Running kinase inhibition data set: 
1. cd Kinase
2. python simulation.py 
### Running 10 clients MNIST
The 10 clients MNIST can be run by running the required Server in MNIST\10Clients as well as running the 10 respective clients. All with a port number. 

1. cd MNIST\10clients

Then choose respective aggregation algorithm. For FedAvg: 

2. python ServerFedAvg.py 5002

Lastly run the 10 clients of the respective distribution. For IID and equal distribution: 

3. cd ClientsIIDED 
4. python client1.py 5002
5. python client2.py 5002
6. python client3.py 5002
7. python client4.py 5002
8. python client5.py 5002
9. python client6.py 5002
10. python client7.py 5002
11. python client8.py 5002
12. python client9.py 5002
13. python client10.py 5002

    
### Running 50 clients MNIST
For IID data distribution: 

1. cd MNIST\50clients
2. python multipleiid.py

For non-IID data distribution: 

1. cd MNIST\50clients
2. python multipleniid.py
