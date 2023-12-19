#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:01:00
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mem-per-cpu=2G


module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/envs/flower2en



srun --ntasks 10 --cpus-per-task=1 --mem-per-cpu=10M python print_id.py &
