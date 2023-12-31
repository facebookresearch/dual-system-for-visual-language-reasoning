#!/bin/bash

home_dir=$1

ckpt_dir="${home_dir}/llama/llama-2-70b"
model_name='llama2-70B'
tokenizer_path="${home_dir}/llama/tokenizer.model"
num_gpu=8

dataset="chartQA"
prompt="cot_5shot"
table_path="${home_dir}/chart2table/outputs/chartQA-test/google/deplot_vqa/inference_all.jsonl"

# sampling 
top_k=0
top_p=1.0
temperature=0.4
num_beams=1
eval_batch_size=1

splits=("test_augmented" "test_human")
for eval_split in ${splits[@]};
do
    data_path="${home_dir}/datasets/ChartQA/Dataset/test/${eval_split}.jsonl"
    output_prefix="${home_dir}/outputs/${dataset}-${eval_split}/deplot_${prompt}_topK${top_k}_topP${top_p}_temp${temperature}_beam${num_beams}_${model_name}"
    mkdir -p $output_prefix
    torchrun --nproc_per_node ${num_gpu} \
        llama_prompting_tableQA_gpt.py \
        --data_path $data_path \
        --table_path $table_path \
        --ckpt_dir $ckpt_dir \
        --tokenizer_path $tokenizer_path \
        --model_name $model_name \
        --output_prefix $output_prefix \
        --dataset $dataset \
        --prompt $prompt \
        --eval_split $eval_split \
        --temperature $temperature \
        --top_k $top_k \
        --top_p $top_p \
        --num_beams $num_beams \
        --eval_batch_size $eval_batch_size \
    >${output_prefix}/log.txt 2>&1 &
done
