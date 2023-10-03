# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import (
    set_seed,
    AutoProcessor,
    Pix2StructForConditionalGeneration,
)

from data_helper_for_decoderQA import (
    load_raw_dataset,
    load_ImageQA_dataset,
    Data_Collator_for_inference,
)


def generate(split, dataloader, processor, model, args):

    model.eval()
    epoch_iterator = tqdm(dataloader, desc="Generate Iteration")

    output_path = os.path.join(
        args.save_dir, "generation_{}_len{}.txt".format(split, args.max_dec_length)
    )
    generated_texts = []
    with open(output_path, "w", buffering=1) as fw:
        for _, batch in enumerate(epoch_iterator):
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
                gen_text = processor.decode(
                    gen[input_length + 1 :], skip_special_tokens=True
                ).strip()
                fw.write(gen_text + "\n")
                generated_texts.append(gen_text)
    return generated_texts


def parse_desc_answer(input):
    input_spans = input.split(". The x-axis shows:")
    if "The figure shows the data of: " in input_spans[0]:
        legend_span = input_spans[0].split("The figure shows the data of: ")[1]
        legend_output = set([span.strip() for span in legend_span.split(" | ")])
    else:
        legend_output = set()
    if len(input_spans) > 1:
        xaxis_span = input_spans[1]
        if xaxis_span.endswith("."):
            xaxis_span = xaxis_span[:-1]
        xaxis_output = set([span.strip() for span in xaxis_span.split(" | ")])
    else:
        xaxis_output = set()
    return legend_output, xaxis_output


def parse_row_column_answer(input):
    if input.endswith("."):
        input = input[:-1]
    if "The data is " in input:
        answer = input.split("The data is ")[1]
        output = set([span.strip() for span in answer.split(",")])
    else:
        output = set()
    return output


def parse_element_answer(input):
    if input.endswith("."):
        input = input[:-1]
    if "The data is " in input:
        answer = input.split("The data is ")[1]
        output = answer.strip()
    else:
        output = ""
    return output


def main(args, seed):
    # ----------------------------------------------------- #
    # model
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=os.path.join(args.home_dir, "hg_cache")
    )
    processor.image_processor.is_vqa = False
    model = Pix2StructForConditionalGeneration.from_pretrained(args.save_dir)
    model.to(args.device)

    # ----------------------------------------------------- #
    # data
    evaluation_result = {}
    for split in ["test"]:
        dataset = load_ImageQA_dataset(split, processor, args)
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=Data_Collator_for_inference(processor, args),
            batch_size=args.eval_batch_size,
        )
        generations = generate(split, data_loader, processor, model, args)

        examples = load_raw_dataset(split, args)
        assert len(examples) == len(generations)
        accuracy_by_type = {"row/column": [], "element": [], "desc": []}
        for example, generation in zip(examples, generations):
            if example["type"] == "header" or example["type"] == "desc":
                legend_answer, xaxis_answer = parse_desc_answer(example["answer"])
                legend_prediction, xaxis_prediction = parse_desc_answer(generation)
                if len(legend_answer) != 0 and len(xaxis_answer) != 0:
                    legend_acc = (
                        len(legend_answer.intersection(legend_prediction))
                        * 1.0
                        / len(legend_answer)
                    )
                    xaxis_acc = (
                        len(xaxis_answer.intersection(xaxis_prediction))
                        * 1.0
                        / len(xaxis_answer)
                    )
                    acc = (legend_acc + xaxis_acc) / 2.0
                    accuracy_by_type["desc"].append(acc)
            elif example["type"] == "row/column":
                answer = parse_row_column_answer(example["answer"])
                prediction = parse_row_column_answer(generation)
                if len(answer) != 0:
                    acc = len(answer.intersection(prediction)) * 1.0 / len(answer)
                    accuracy_by_type["row/column"].append(acc)
            else:  # example["type"] == "element":
                assert example["type"] == "element", example["type"]
                answer = parse_element_answer(example["answer"])
                prediction = parse_element_answer(generation)
                if answer == prediction:
                    accuracy_by_type["element"].append(1.0)
                else:
                    accuracy_by_type["element"].append(0.0)

        evaluation_result[split] = {}
        accuracy = 0.0
        for type in accuracy_by_type:
            accuracy_by_type[type] = (
                sum(accuracy_by_type[type]) * 100.0 / len(accuracy_by_type[type])
            )
            evaluation_result[split][type] = accuracy_by_type[type]
            accuracy += accuracy_by_type[type]
        evaluation_result[split]["overall_accuracy"] = accuracy / len(accuracy_by_type)

    result_file = os.path.join(
        args.save_dir, "evaluation_len{}.json".format(args.max_dec_length)
    )
    with open(result_file, "w") as fw:
        json.dump(evaluation_result, fw, indent=4)

    print("Evaluation done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main.")
    parser.add_argument("--home_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--image_dir_chartQA", type=str)
    parser.add_argument("--image_dir_plotQA", type=str)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--save_dir", "-o", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")

    # model
    parser.add_argument("--model_name", "-m", type=str, default="google/deplot")
    parser.add_argument("--max_enc_length", type=int, default=128)
    parser.add_argument("--max_dec_length", type=int, default=512)
    parser.add_argument("--max_patches", type=int, default=2048)

    # training
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--grad_step", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_epoch", type=float, default=1000)
    parser.add_argument("--num_epoch_early_stopping", type=int, default=10)
    parser.add_argument("--num_training_steps", type=int, default=10000)

    # inference
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--overwrite_output", action="store_true")

    # gpu and workers option
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.gpu))
    args.image_dirs = {
        "chartQA": args.image_dir_chartQA,
        "plotQA": args.image_dir_plotQA,
    }

    seed = 42
    set_seed(seed)
    main(args, seed)
