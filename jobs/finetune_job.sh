#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=gpu
#SBATCH --nodelist=wn202,wn203,wn204,wn205,wn206,wn207
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err

echo "Starting fine tuning..."

# Display the hostname and GPU info
echo "GPU info on node $HOSTNAME"
nvidia-smi

# Test GPU visibility
singularity exec --nv containers/finetune.sif python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0))"

# Export the variable so it's visible inside the container environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your inference script
singularity exec --nv containers/finetune.sif python3 ~/NLP/src/cot_finetune.py
