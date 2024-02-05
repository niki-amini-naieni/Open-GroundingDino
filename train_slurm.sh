PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
CFG=$3
DATASETS=$4
OUTPUT_DIR=$5
export TOKENIZERS_PARALLELISM=true

srun -p ${PARTITION} \
    --job-name=open_G_dino \
    --gpus=rtx8k:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u main.py --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path /scratch/shared/beegfs/nikian/gdino_weights/gdinot-1.8m-odvg.pth \
        --options text_encoder_type=/scratch/shared/beegfs/nikian/bert-base-uncased
