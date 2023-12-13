#!/bin/bash
nc=2
nro=10
str=5
lr=0.1
rho=0.5
dset="MNIST_10c"

while true;
do
    case "$1" in
        --nc)   nc=$2; shift 2;;
        --nro) nro=$2; shift 2;;
        --str)  str=$2; shift 2;;
        --lr) lr=$2; shift 2;;
        --rho) rho=$2; shift 2;;
        --dset) dset=$2; shift 2;;
        --) shift;  break   ;;
        *)  break ;;
    esac
done


python run_server.py --str $str --nro $nro --nc $nc --rho $rho --dset $dset &

sleep 10

for ((i = 0; i<=$nc-1; i++))
do 
    sleep 3
    python run_client.py --cid $i --lr ${lr} --rho ${rho} --dset ${dset} &
done

wait $!


exit
