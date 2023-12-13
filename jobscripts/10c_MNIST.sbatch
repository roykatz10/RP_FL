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
module load miniconda/3.9
conda activate /tudelft.net./staff-bulk/ewi/insy/DL/swiergarst/envs/flower2-en


srun bash run.sh --nc 10 --nro 5 --dset=MNIST_10c