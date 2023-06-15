#!/bin/bash


ckpt_dir='/large_experiments/fair_llm/genesis/consolidated_ckpts/30B_1.4T_consolidated_fp16_mp4'
model_name='llama-30B'
num_gpu=4
# ckpt_dir='/large_experiments/fair_llm/genesis/consolidated_ckpts/70B_1.4T_consolidated_fp16_mp8'
# model_name='llama-70B'
# num_gpu=8
tokenizer_path='/checkpoint/kshuster/projects/genesis/tokenizer_final_32k.minus_inf_ws.model'

dataset="chartQA"
prompt="cot_7shot_V6"
table_model="deplot_vqa"
table_path="../chart2table/outputs/chartQA-test/google/deplot_vqa/inference_all.jsonl"

# sampling 
top_k=0
top_p=1.0
temperature=1.0
num_beams=1
eval_batch_size=1

splits=("test_augmented" "test_human")
for eval_split in ${splits[@]};
do
    data_path="../datasets/ChartQA/Dataset/test/${eval_split}.json"
    output_prefix="outputs/${dataset}-${eval_split}/${table_model}-${prompt}_topK${top_k}_topP${top_p}_temp${temperature}_beam${num_beams}.${model_name}"
    mkdir -p $output_prefix
    srun --partition=learnfair --constraint=volta32gb --gres=gpu:volta:${num_gpu} --time 2-00:00 --ntasks-per-node=1 --cpus-per-task=10 --mem=400G torchrun --nproc_per_node ${num_gpu} \
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
    >${output_prefix}/log.txt 2>&1 &
done
