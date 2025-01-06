#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH -a 1-10:1
#$ -N FM_data

source /etc/profile

module add cuda/11.2
module add anaconda3

source activate tf-gpu

python CAL/Experiments/FM/FM.py
