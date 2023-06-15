#!/bin/bash

#SBATCH --job-name=vlqa-multinode
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=40
#SBATCH --mem=400GB
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/error_%j.txt
#SBATCH --partition=learnfair
#SBATCH --time=2-00:00
#SBATCH --constraint=volta32gb

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

num_gpu=8
num_nodes=8
image_dir="../datasets/vqa-2/images/train_val_2014"
dataset="vqa-2_caption"
model_name='google/deplot'
max_dec_length=512
train_batch_size=256
learning_rate=1e-5
weight_decay=0
num_training_steps=20000

save_dir="./checkpoints/${dataset}/pad_decoderQA_${model_name}_bs${train_batch_size}_gs${grad_step}_ngpu${num_gpu}_node${num_nodes}_lr${learning_rate}_wd${weight_decay}_step${num_training_steps}"
mkdir -p $save_dir
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

resume_from_checkpoint=$1

srun torchrun --nnodes $num_nodes --nproc_per_node $num_gpu --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29700 \
    main_trainer.py \
    --resume_from_checkpoint $resume_from_checkpoint \
    --image_dir $image_dir \
    --dataset $dataset \
    --save_dir $save_dir \
    --model_name $model_name \
    --max_dec_length $max_dec_length \
    --train_batch_size $train_batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_training_steps $num_training_steps \
    | tee ${save_dir}/debug.log