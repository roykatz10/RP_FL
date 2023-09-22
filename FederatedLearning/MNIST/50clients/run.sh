#!/bin/bash
nc=50
nro=10
str=0

while true;
do
    case "$1" in
        --nc)   nc=$2; shift 2;;
        --str)  str=$2; shift 2;;
        --) shift;  break   ;;
        *)  break ;;
    esac
done


python run_server.py --str $str --nro $nro &

for ((i = 0; i<=$nc-1; i++))
do 
    sleep 1
    python run_client.py --cid $i &
done
wait $!


exit