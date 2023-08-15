#!/bin/bash

home_dir="/private/home/peifengw"

model_name='finetuned_llama2_cot_replace100_seed42'
ckpt_dir="${home_dir}/checkpoints/finetuned_llama2_fixseed/70B_chartplot_replace100_seed42"
# model_name='llama2-70B'
# ckpt_dir="${home_dir}/llama/llama-2-70b"

tokenizer_path="${home_dir}/llama/tokenizer.model"
num_gpu=8

dataset="figureQA"
prompt="cot_5shot"

# vlqa model
vlqa_name='google/deplot'
vlqa_dir="${home_dir}/checkpoints/finetuned_deplot/"
max_vlqa_len=512

# sampling 
top_k=0
top_p=1.0
temperature=1.0
num_beams=1
eval_batch_size=1

eval_splits=("val1_sample5K" "val2_sample5K")
image_dirs=("${home_dir}/datasets/figureQA/validation1/png/" "${home_dir}/datasets/figureQA/validation2/png/")
split_idx=0
num_process=2
for eval_split in ${eval_splits[@]};
do
    for ((split=0; split<$num_process; split++));
    do
        data_path="${home_dir}/datasets/figureQA/${eval_split}.jsonl"
        image_dir=${image_dirs[split_idx]}
        output_prefix="${home_dir}/outputs/${dataset}-${eval_split}/dual_${model_name}_${prompt}_topK${top_k}_topP${top_p}_temp${temperature}_beam${num_beams}"
        mkdir -p $output_prefix
        srun --partition=learnfair --constraint=volta32gb --gres=gpu:volta:${num_gpu} --time 2-00:00 --ntasks-per-node=1 --cpus-per-task=40 --mem=400G torchrun --nproc_per_node ${num_gpu} --master_port 302${split_idx}${split} \
            llama_prompting_dualsys.py \
            --home_dir $home_dir \
            --data_path $data_path \
            --image_dir $image_dir\
            --ckpt_dir $ckpt_dir \
            --tokenizer_path $tokenizer_path \
            --model_name $model_name \
            --vlqa_name $vlqa_name \
            --vlqa_dir $vlqa_dir \
            --max_vlqa_len $max_vlqa_len \
            --output_prefix $output_prefix \
            --dataset $dataset \
            --prompt $prompt \
            --eval_split $eval_split \
            --temperature $temperature \
            --top_k $top_k \
            --top_p $top_p \
            --num_beams $num_beams \
            --eval_batch_size $eval_batch_size \
            --num_process $num_process \
            --split $split \
        >${output_prefix}/eval_log_split${split}-${num_process}.txt 2>&1 &
    done
    ((split_idx++))
done
