#!/bin/bash


ckpt_dir='/large_experiments/fair_llm/genesis/consolidated_ckpts/30B_1.4T_consolidated_fp16_mp4'
model_name='llama-30B'
num_gpu=4
# ckpt_dir='/large_experiments/fair_llm/genesis/consolidated_ckpts/70B_1.4T_consolidated_fp16_mp8'
# model_name='llama-70B'
# num_gpu=8
tokenizer_path='/checkpoint/kshuster/projects/genesis/tokenizer_final_32k.minus_inf_ws.model'

dataset="a-okvqa"
image_dir="../datasets/aokvqa/images/val2017/"

# vlqa model
vlqa_name='google/deplot'
vlqa_dir=$1
vlqa_mode="decoderQA"
max_vlqa_len=512

# sampling 
top_k=0
top_p=1.0
temperature=1.0
num_beams=1
eval_batch_size=1

splits=("val_DA" "val_MC")
prompts=("cot_5shot_V0_caption_DA" "cot_5shot_V0_caption_MC")
split_idx=0
num_process=1
for eval_split in ${splits[@]};
do
    prompt=${prompts[split_idx]}
    for ((split=0; split<$num_process; split++));
    do
        data_path="../datasets/aokvqa/${eval_split}.jsonl"
        output_prefix="${vlqa_dir}/${dataset}-${eval_split}_${model_name}-${prompt}_topK${top_k}_topP${top_p}_temp${temperature}_beam${num_beams}"
        mkdir -p $output_prefix
        srun --partition=learnfair --constraint=volta32gb --gres=gpu:volta:${num_gpu} --time 2-00:00 --ntasks-per-node=1 --cpus-per-task=40 --mem=400G torchrun --nproc_per_node ${num_gpu} --master_port 298${split_idx}${split} \
            llama_vqa.py \
            --data_path $data_path \
            --image_dir $image_dir\
            --ckpt_dir $ckpt_dir \
            --tokenizer_path $tokenizer_path \
            --model_name $model_name \
            --vlqa_name $vlqa_name \
            --vlqa_dir $vlqa_dir \
            --vlqa_mode $vlqa_mode \
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
        >${output_prefix}/log_split${split}-${num_process}.txt 2>&1 &
    done
    ((split_idx++))
done
