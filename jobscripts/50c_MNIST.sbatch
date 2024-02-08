#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=8:00:00
#SBATCH --ntasks=51
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mem-per-cpu=2G
#SBATCH 

module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/flower


nc=50
nro=400
str=0
lr=0.0001
rho=0.5
dset="MNIST_50c"
iid=0
ed=0

filename="log_nc${nc}_nro${nro}_str${str}_lr${lr}_rho${rho}_dset${dset}_iid${iid}_ed{$ed}.txt"
line_tocheck='Flower ECE: gRPC server running'

cd ..

echo "$filename"

# remove output file if it already exists
if test -f "$filename"; then
  rm "$filename"
fi


srun  --ntasks 1 --cpus-per-task=1 --exclusive --mem-per-cpu=2G python run_server.py --str $str --nro $nro --nc $nc --rho $rho --dset $dset --fn $filename &

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


srun --ntasks $nc --cpus-per-task=1 --exclusive --mem-per-cpu=2G python run_client.py --lr $lr --rho $rho --dset $dset --iid $iid --ed $ed --str $str &

wait $!

rm "$filename"



