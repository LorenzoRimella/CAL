#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#$ -N FM_SMC_ovd
#SBATCH -a 1-10:1

source /etc/profile

cd $global_storage

module add anaconda3/2023.09
module add cuda/12.5

source activate tf-gpu

python CAL/Experiments/FM/FM_ovd_rep.py
