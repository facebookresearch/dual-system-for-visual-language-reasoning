#!/bin/bash

#SBATCH --job-name=vlqa-multinode
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=400GB
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/error_%j.txt
#SBATCH --time=2-00:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

home_dir=$1

num_gpu=8
num_nodes=8
image_dir_chartQA="${home_dir}/datasets/ChartQA/Dataset/train/png/"
image_dir_plotQA="${home_dir}/datasets/PlotQA/png/train/"
dataset="${home_dir}/datasets/atomic_chartqa"
model_name='google/deplot'
max_dec_length=512
train_batch_size=256
learning_rate=1e-5
weight_decay=0
num_training_steps=10000

save_dir="${home_dir}/outputs/checkpoints/finetuned_${model_name}_bs${train_batch_size}_gs${grad_step}_ngpu${num_gpu}_node${num_nodes}_lr${learning_rate}_wd${weight_decay}_step${num_training_steps}"
mkdir -p $save_dir
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun --nnodes $num_nodes --nproc_per_node $num_gpu --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29700 \
    main_trainer.py \
    --image_dir_chartQA $image_dir_chartQA \
    --image_dir_plotQA $image_dir_plotQA \
    --dataset $dataset \
    --save_dir $save_dir \
    --model_name $model_name \
    --max_dec_length $max_dec_length \
    --train_batch_size $train_batch_size \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_training_steps $num_training_steps \
    | tee ${save_dir}/debug.log