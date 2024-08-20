#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --output=/itet-stor/maxihuber/net_scratch/runs/%j/train_%j_%t.out
#SBATCH --error=/itet-stor/maxihuber/net_scratch/runs/%j/train_%j_%t.err
#SBATCH --open-mode=append
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=10
#SBATCH --mem=1250G
#SBATCH --nodelist=tikgpu10

# More options, requeue hasn't worked so far however (i do it manually)
## SBATCH --time=00:10:00
## SBATCH --requeue
## SBATCH --signal=B:USR1@90
## SBATCH --signal=TERM@90

# Create directories for job
mkdir -p /itet-stor/maxihuber/net_scratch/runs/$SLURM_JOB_ID
mkdir -p /itet-stor/maxihuber/net_scratch/runs/$SLURM_JOB_ID/metrics

# Exit on errors
set -o errexit

# Initialize Conda environment
source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate fastenv

# Navigate to your project directory (if necessary)
cd /home/maxihuber/eeg-foundation/

# Send some noteworthy information to the output log
for node in $(scontrol show hostname $SLURM_JOB_NODELIST); do
    echo "Running on node: $node"
    echo "In directory:    $(pwd)"
    echo "Starting on:     $(date)"
    echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_NODES: ${SLURM_JOB_NUM_NODES}"
done

# Trap SIGUSR1 signal for requeueing
# Trap SIGUSR1 and SIGTERM signals for requeueing
# trap 'echo "Caught SIGUSR1 signal! Requeuing..."; scontrol requeue $SLURM_JOB_ID' USR1
# trap 'echo "Caught SIGTERM signal! Requeuing..."; scontrol requeue $SLURM_JOB_ID' TERM

# Make sure wandb recognizes us
wandb login
srun python /home/maxihuber/eeg-foundation/src/train2.py experiment=rope.yaml
# srun python /home/maxihuber/eeg-foundation/src/val.py experiment=rope.yaml

# Start training with or without checkpoint
# ROUTINE_CHECKPOINT_DIR="/itet-stor/maxihuber/net_scratch/checkpoints/${SLURM_JOB_ID}/routine"
# if [ -d "$ROUTINE_CHECKPOINT_DIR" ]; then
#     echo "Restoring from checkpoint."
#     srun python /home/maxihuber/eeg-foundation/src/train.py experiment=rope.yaml restore_from_checkpoint=True
# else
#     echo "Starting without checkpoint."
#     srun python /home/maxihuber/eeg-foundation/src/train.py experiment=rope.yaml
# fi

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
