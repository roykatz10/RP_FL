import time
import sys


time.sleep(3)
stdout_orig = sys.stdout


sys.stdout = open("output.txt", "w")


print("bla bla bla Flower ECE: gRPC server running or something")


sys.stdout.close()
sys.stdout = stdout_orig


while(True):
  time.sleep(1)

