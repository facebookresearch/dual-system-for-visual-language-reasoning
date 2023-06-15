#!/bin/bash

image_dir="../datasets/vqa-2/images/train_val_2014"
dataset="vqa-2"
model_name='google/deplot'
max_dec_length=512
eval_batch_size=1

num_gpu=1

save_dir=$1

# python \
srun --partition=learnfair --constraint=volta32gb --gres=gpu:volta:${num_gpu} --time 2-00:00 --ntasks-per-node=1 --cpus-per-task=10 --mem=400G torchrun --nproc_per_node ${num_gpu} \
    evaluation_vqa.py \
    --image_dir $image_dir \
    --dataset $dataset \
    --save_dir $save_dir \
    --model_name $model_name \
    --max_dec_length $max_dec_length \
    --eval_batch_size $eval_batch_size \
    > ${save_dir}/inference_len${max_dec_length}.log 2>&1 &