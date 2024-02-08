#!/bin/bash
nc=2
nro=10
str=0
lr=0.001
rho=0.5
dset="MNIST_10c"
central=0

while true;
do
    case "$1" in
        --nc)   nc=$2; shift 2;;
        --nro) nro=$2; shift 2;;
        --str)  str=$2; shift 2;;
        --lr) lr=$2; shift 2;;
        --rho) rho=$2; shift 2;;
        --dset) dset=$2; shift 2;;
        --central) central=1; nc=1; shift 1;;
        --) shift;  break   ;;
        *)  break ;;
    esac
done


filename="log_nc${nc}_nro${nro}_str${str}_lr${lr}_rho${rho}_dset${dset}_iid${iid}_ed${ed}.txt"
line_tocheck='Flower ECE: gRPC server running'

# first, kill al other processes if they are still going on
pkill -9 -f run_server.py
pkill -9 -f run_client.py

# remove output file if it already exists
if test -f "$filename"; then
  rm "$filename"
fi

echo "central $central"

# startup server
python run_server.py --str $str --nro $nro --nc $nc --rho $rho --dset $dset --fn $filename --central $central & 

# whole bunch of stuff to make sure we're waiting for the server over here
finished=0
while [ $finished -eq 0 ] 
do
  # check if the output file exists
  if test -f "$filename"; then
    #read the file
    while IFS= read -r line
    do
      if [[ $line == *$line_tocheck* ]]; then
        echo "server booted up. starting clients..."
        finished=1
        break
      fi
    done < "$filename"
  fi
done



for ((i = 0; i<=$nc-1; i++))
do 
    sleep 1
    python run_client.py --cid $i --lr ${lr} --rho ${rho} --dset ${dset} --str $str --central $central &
done

wait $!

rm "$filename"


exit
