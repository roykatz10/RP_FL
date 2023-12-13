#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --mem=14000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END






module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/envs/flower2en

cd ..

srun bash run.sh --nc 2 --nro 3 --dset=MNIST_10c
