#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p res-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --qos=short
#SBATCH -t 00-15:00:00
#SBATCH --job-name=pysindy-ana
#SBATCH --mem=16G

#SBATCH -e stderr-filename
#SBATCH --mail-type=ALL
#SBATCH --mail-user qtzk83@durham.ac.uk

# module purge
# Load CUDA specific version to match environment.yml
module load cuda/12.4

# Initialize conda
# source /home2/qtzk83/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
source activate pyargos-shred-dev

# Add these diagnostic lines
echo "=== Environment Information ==="
echo "Modules loaded:"
module list
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

echo "=== GPU Information ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"
echo ""

# Try different GPU detection methods
echo "Method 1: nvidia-smi"
nvidia-smi || echo "nvidia-smi failed"
echo ""

echo "Method 2: nvidia-debugdump"
nvidia-debugdump -l || echo "nvidia-debugdump failed"
echo "======================"

# Run the script 
./run_analysis_ba_script.sh