#!/bin/bash
nc=3
nro=10
str=5
lr=0.1
rho=0.5

while true;
do
    case "$1" in
        --nc)   nc=$2; shift 2;;
        --str)  str=$2; shift 2;;
        --) shift;  break   ;;
        *)  break ;;
    esac
done


python run_server.py --str $str --nro $nro --nc $nc --rho $rho &



for ((i = 0; i<=$nc-1; i++))
do 
    sleep 1
    python run_client.py --cid $i --lr ${lr} --rho ${rho} &
done

wait $!


exit