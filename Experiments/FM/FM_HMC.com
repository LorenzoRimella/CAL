#!/bin/bash
#SBATCH -p gpu-long
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#$ -N CAL_FM_HMC

source /etc/profile

cd $global_storage

module add anaconda3/2023.09
module add cuda/12.5

python CAL/Experiments/FM/FM_HMC.py
