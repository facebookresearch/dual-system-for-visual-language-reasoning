import json
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import set_seed, AutoProcessor, Pix2StructForConditionalGeneration, get_cosine_schedule_with_warmup 
from transformers.optimization import Adafactor

from data_helper_for_vqa import load_raw_dataset, load_ImageQA_dataset, Data_Collator_for_inference

def evaluate(dataloader, model, args):

    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Eval Iteration")

    loss_sum = 0.
    ppl_sum = 0.
    tokens_sum = 0.
    for step, batch in enumerate(epoch_iterator):
        labels = batch.pop("labels").to(args.device)
        flattened_patches = batch.pop("flattened_patches").to(args.device)
        attention_mask = batch.pop("attention_mask").to(args.device)
        with torch.no_grad():
            outputs = model(
                        flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            tokens_sum += num_tokens
            ppl_sum += outputs.loss.item() * num_tokens

            loss_sum += loss.item()
        if args.debug and step > 10:
            break

    loss_sum /= (step + 1)
    ppl_sum = math.exp(ppl_sum / tokens_sum)

    return {"loss": loss_sum, "perplexity": ppl_sum}

def generate(split, dataloader, processor, model, args):

    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Generate Iteration")

    output_path = os.path.join(args.save_dir, 'generation_{}_len{}.txt'.format(split, args.max_dec_length))
    generated_texts = []
    with open(output_path, 'w', buffering=1) as fw:
        for step, batch in enumerate(epoch_iterator):
            flattened_patches = batch.pop("flattened_patches").to(args.device)
            attention_mask = batch.pop("attention_mask").to(args.device)
            decoder_input_ids = batch.pop("decoder_input_ids").to(args.device)
            input_length = len(decoder_input_ids[0])
            generations = model.generate(
                        decoder_input_ids=decoder_input_ids,
                        flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_dec_length,
                    )
            for gen in generations:
                gen_text = processor.decode(gen[input_length+1:], skip_special_tokens=True).strip()
                fw.write(gen_text + "\n")
                generated_texts.append(gen_text)
    return generated_texts 

def main(args, seed):
    # ----------------------------------------------------- #
    # model
    processor = AutoProcessor.from_pretrained(args.model_name, cache_dir='../hg_cache')
    processor.image_processor.is_vqa = False
    model = Pix2StructForConditionalGeneration.from_pretrained(args.save_dir)
    model.to(args.device)

    # ----------------------------------------------------- #
    # data
    evaluation_result = {}
    for split in ['test']:
        dataset = load_ImageQA_dataset(split, processor, args)
        data_loader = DataLoader(dataset,
                    shuffle=False,
                    collate_fn=Data_Collator_for_inference(processor, args),
                    batch_size=args.eval_batch_size,
        )
        generations = generate(split, data_loader, processor, model, args)
        # exact accuracy

        # with open(os.path.join(args.save_dir, 'generation_{}.txt'.format(split, args.max_dec_length)), 'r') as fr:
        #     generations = [line.strip() for line in fr.readlines()]
        examples = load_raw_dataset(split, args)
        assert len(examples) == len(generations)
        accuracy = 0.
        accuracy_by_type = defaultdict(list)
        debug_file = open(os.path.join(args.save_dir, 'debug_{}.jsonl'.format(split, args.max_dec_length)), 'w')
        for example, generation in zip(examples, generations):
            if example["answer"] == generation:
                accuracy += 1
                accuracy_by_type[example["type"]].append(1.)
            else:
                accuracy_by_type[example["type"]].append(0.)
            if not example["answer"] == generation:
                debug_file.write(json.dumps({"id": example["question_id"], "answer": example["answer"], "prediction": generation, "question": example["question"]}) + "\n")

        accuracy /= len(examples)
        evaluation_result[split] = {"overall accuracy": accuracy * 100.}
        for type in accuracy_by_type:
            accuracy_by_type[type] = sum(accuracy_by_type[type]) * 100. / len(accuracy_by_type[type])
            evaluation_result[split][type] = accuracy_by_type[type] 
        
        debug_file.close()
    result_file = os.path.join(args.save_dir, 'evaluation_{}_len{}.json'.format(args.dataset, args.max_dec_length))
    with open(result_file, 'w') as fw:
        json.dump(evaluation_result, fw, indent=4)

    print("Evaluation done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--save_dir', '-o', type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--save_ckpt", action='store_true')

    # model
    parser.add_argument('--model_name', '-m', type=str)
    parser.add_argument('--max_enc_length', type=int, default=128)
    parser.add_argument('--max_dec_length', type=int, default=128)
    parser.add_argument('--max_patches', type=int, default=2048)

    # training
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--num_epoch', type=float, default=1000)
    parser.add_argument('--num_epoch_early_stopping', type=int, default=10)
    parser.add_argument('--num_training_steps', type=int, default=10000)

    # inference
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument("--overwrite_output", action='store_true')

    # gpu and workers option
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    seed = 42
    set_seed(seed)
    main(args, seed)
