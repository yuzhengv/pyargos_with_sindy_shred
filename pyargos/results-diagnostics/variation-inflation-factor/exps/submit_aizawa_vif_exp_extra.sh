#!/bin/bash

#SBATCH -p shared
#SBATCH -c 1
#SBATCH -n 28
##SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=aizawa_vif_exp_extra

# Load Application
# module purge
# module load r/4.2.1
# module load python/3.9.9

source activate pyargos-shred-dev
    
python aizawa_vif_exp_extra.py