# RP_FL
Aggregation-algorithm-fl-comparison\

This project requires flower and can be installed by:\
!pip install -q flwr[simulation] torch torchvision matplotlib\


The kinase inhibition dataset can be run by:
cd Kinase
python simulation.py


The 10 clients MNIST can be run by running the required Server in MNIST\10Clients as well as running the 10 respective clients. All with a port number.\
cd 10clients \
python ServerFedAvg.py 5002\
python client1.py 5002\


The 50 clients MNIST can be run with:\
cd 50clients\
python multipleiid.py\

or\
cd 50clients\
python multipleniid.py\
