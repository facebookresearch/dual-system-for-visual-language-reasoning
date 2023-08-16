#!/bin/bash

home_dir="/private/home/peifengw"

image_dir_chartQA="${home_dir}/datasets/ChartQA/Dataset/train/png/"
image_dir_plotQA="${home_dir}/datasets/PlotQA/png/train/"
dataset="${home_dir}/datasets/atomic_chartqa"
model_name='google/deplot'
max_dec_length=512
eval_batch_size=1

num_gpu=1

save_dir=$1

srun --partition=learnfair --constraint=volta32gb --gres=gpu:volta:${num_gpu} --time 2-00:00 --ntasks-per-node=1 --cpus-per-task=10 --mem=400G torchrun --nproc_per_node ${num_gpu} python \
    evaluation_atomicQA.py \
    --image_dir_chartQA $image_dir_chartQA \
    --image_dir_plotQA $image_dir_plotQA \
    --dataset $dataset \
    --save_dir $save_dir \
    --model_name $model_name \
    --max_dec_length $max_dec_length \
    --eval_batch_size $eval_batch_size \
    > ${save_dir}/inference_len${max_dec_length}.log 2>&1 &

