#!/bin/bash

#SBATCH --job-name=qwen_nlp
#SBATCH --qos=cscc-gpu-qos
#SBATCH --partition=cscc-gpu-p                     # queue name
#SBATCH --mail-type=all                      # mail events (none, begin, end, fail, all)
#SBATCH --mail-user=nurkhan.laiyk@mbzuai.ac.ae   # where to send mail
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100000                         # job memory request in megabytes
#SBATCH --gres=gpu:1                          # number of gpus
#SBATCH --time=12:00:00                   # time limit hrs:min:sec or dd-hrs:min:sec
#SBATCH --output=/l/users/nurkhan.laiyk/nlp-project/output_%j.log   # Standard output log
#SBATCH --error=/l/users/nurkhan.laiyk/nlp-project/error_%j.log    # Error log


# Activate Conda environment
source activate /home/nurkhan.laiyk/miniconda3/envs/unsloth_env

# Change directory to the project folder
cd /home/nurkhan.laiyk/nlp-project


python qwen_inference.py
