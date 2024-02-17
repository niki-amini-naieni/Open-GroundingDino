#!/bin/bash
#SBATCH --job-name=finetune_gdino_fsc147_lr_1e-4_thresh_0.23_vit_b_odvg_finetune_image_encoder
#SBATCH --ntasks=1 --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=rtx8k:1
#SBATCH --output=/users/nikian/Open-GroundingDino/train-log-gdino-lr-1e-4-thresh-0.23-vit-b-12-2-2024-finetune-image-encoder.txt

export CUDA_VISIBLE_DEVICES=0
#export TOKENIZERS_PARALLELISM=true
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain.
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

unzip /scratch/shared/beegfs/nikian/FSC-147.zip -d $TMPDIR
cp -r /scratch/shared/beegfs/nikian/fsc147_odvg $TMPDIR
cp -r /scratch/shared/beegfs/nikian/fsc147_coco $TMPDIR

nohup python -m torch.distributed.launch  --nproc_per_node=1 main.py \
        --output_dir ./logs_12_2_2024_thresh_0.23_lr_1e-4_vit_b_finetune_image_encoder \
        -c config/cfg_fsc147_vit_b_odvg.py \
        --datasets config/datasets_fsc147.json  \
        --pretrain_model_path /scratch/shared/beegfs/nikian/gdino_weights/groundingdino_swinb_cogcoor.pth \
        --options text_encoder_type=/scratch/shared/beegfs/nikian/bert-base-uncased
