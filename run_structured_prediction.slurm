#!/bin/bash

#SBATCH -J structured_pred
#SBATCH -A r01662
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o structured_out_%j.log
#SBATCH -e structured_err_%j.err

# Load Python module (use a recent one)
module load python/3.10.10

# Activate your virtualenv (update if needed)
source ~/pathoenv/bin/activate

# Go to the repo
cd /N/project/wsiclass/HANCOCK_MultimodalDataset

# Run the script
python multimodal_machine_learning/execution/outcome_prediction.py \
  /N/project/wsiclass/Hancock_Dataset/DataSplits_DataDictionaries \
  /N/project/wsiclass/HANCOCK_MultimodalDataset/features \
  /N/project/wsiclass/HANCOCK_MultimodalDataset/structured \
  recurrence
