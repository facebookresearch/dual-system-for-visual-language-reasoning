#!/bin/bash

home_dir="/private/home/peifengw"

ckpt_dir="${home_dir}/llama/llama-2-70b"
model_name='llama2-70B'
tokenizer_path="${home_dir}/llama/tokenizer.model"
num_gpu=8

dataset="dvQA"
prompt="cot_5shot_dvqa"
table_path="${home_dir}/chart2table/outputs/dvQA-val_hard_qa_reasoning_10K/google/deplot_vqa/inference_all.jsonl"

# sampling 
top_k=0
top_p=1.0
temperature=0.4
num_beams=1
eval_batch_size=1

eval_splits=("val_hard_qa_reasoning_10K")
num_process=1
for eval_split in ${eval_splits[@]};
do
    for ((split=0; split<$num_process; split++));
    do
        data_path="${home_dir}/datasets/dvqa/${eval_split}.jsonl"
        output_prefix="${home_dir}/outputs/${dataset}-${eval_split}/deplot_${prompt}_topK${top_k}_topP${top_p}_temp${temperature}_beam${num_beams}.${model_name}"
        mkdir -p $output_prefix
        srun --partition=learnfair --constraint=volta32gb --gres=gpu:volta:${num_gpu} --time 2-00:00 --ntasks-per-node=1 --cpus-per-task=10 --mem=400G torchrun --nproc_per_node ${num_gpu}  --master_port 2970${split}\
            llama_prompting_tableQA.py \
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
            --num_process $num_process \
            --split $split \
        >${output_prefix}/log_split${split}-${num_process}.txt 2>&1 &
    done
done
