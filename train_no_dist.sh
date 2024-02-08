#!/bin/bash
#SBATCH --job-name=finetune_gdino_fsc147
#SBATCH --ntasks=1 --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=rtx8k:1
#SBATCH --output=/users/nikian/Open-GroundingDino/train-log-gdino.txt
export MASTER_ADDR=localhost
export MASTER_PORT=6006
export CUDA_VISIBLE_DEVICES=0


# Change ``pretrain_model_path`` to use a different pretrain.
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

unzip /scratch/shared/beegfs/nikian/FSC-147.zip -d $TMPDIR
cp -r /scratch/shared/beegfs/nikian/fsc147_odvg $TMPDIR
cp -r /scratch/shared/beegfs/nikian/fsc147_coco $TMPDIR

nohup python main.py \
        --output_dir ./logs \
        -c config/cfg_fsc147.py \
        --datasets config/datasets_fsc147.json  \
        --pretrain_model_path /scratch/shared/beegfs/nikian/gdino_weights/gdinot-1.8m-odvg.pth \
        --options text_encoder_type=/scratch/shared/beegfs/nikian/bert-base-uncased
