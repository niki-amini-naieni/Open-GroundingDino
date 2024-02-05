#!/bin/bash
#SBATCH --job-name=finetune_gdino_fsc147
#SBATCH --ntasks=2 --nodes=1 --ntasks-per-node=2
#SBATCH --mem-per-cpu=24G
#SBATCH --time=48:00:00
#SBATCH --partition=ddp-2way
#SBATCH --gpus=a40:2
#SBATCH --output=/users/nikian/Open-GroundingDino-orig/train-log.txt

export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=true
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain.
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

nohup python -m torch.distributed.launch  --nproc_per_node=2 main.py \
        --output_dir ./logs \
        -c config/cfg_fsc147.py \
        --datasets config/datasets_fsc147.json  \
        --pretrain_model_path /scratch/shared/beegfs/nikian/gdino_weights/gdinot-1.8m-odvg.pth \
        --options text_encoder_type=/scratch/shared/beegfs/nikian/bert-base-uncased
