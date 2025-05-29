#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --nodelist=wn202,wn203,wn204,wn205,wn206,wn207
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:07:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

echo "Starting inference job..."

# Test GPU visibility
singularity exec --nv containers/finetune.sif python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0))"

# Export the variable so it's visible inside the container environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your inference script
singularity exec --nv containers/finetune.sif python3 ~/NLP/src/chat.py
