#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END






module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/envs/flower2en

cd ..

srun bash run.sh --nc 10 --nro 10 --str 0 --dset=MNIST_10c
