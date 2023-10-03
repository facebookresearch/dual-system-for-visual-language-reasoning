# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from transformers import (
    set_seed,
    AutoProcessor,
    Pix2StructForConditionalGeneration,
)

from data_helper_for_decoderQA import load_ImageQA_dataset, Data_Collator


def main(args, seed):
    # ----------------------------------------------------- #
    # params
    gradient_accumulation_steps = args.train_batch_size // args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        if int(os.environ.get("LOCAL_RANK", -1)) > 0:
            sys.stdout = open(os.devnull, "w")

    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=1000,
        max_steps=args.num_training_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        logging_steps=10,
        optim="adamw_hf",
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        output_dir=args.save_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        remove_unused_columns=False,
        log_on_each_node=False,
    )

    # ----------------------------------------------------- #
    # model
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=os.path.join(args.home_dir, "hg_cache")
    )
    processor.image_processor.is_vqa = False
    if args.resume_from_checkpoint:
        print("Resuming from:", args.resume_from_checkpoint)
        model = Pix2StructForConditionalGeneration.from_pretrained(
            args.resume_from_checkpoint
        )
    else:
        model = Pix2StructForConditionalGeneration.from_pretrained(
            args.model_name, cache_dir=os.path.join(args.home_dir, "hg_cache")
        )
    # model.to(args.device)

    # ----------------------------------------------------- #
    # data

    dataset_splits = {}
    for split in ["train", "dev", "test"]:
        dataset_splits[split] = load_ImageQA_dataset(split, processor, args)

    # ----------------------------------------------------- #

    trainer = Trainer(
        model=model,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["dev"],
        args=training_args,
        data_collator=Data_Collator(processor, args),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # ----------------------------------------------------- #
    return_result = {}
    return return_result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main.")
    parser.add_argument("--home_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--image_dir_chartQA", type=str)
    parser.add_argument("--image_dir_plotQA", type=str)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--save_dir", "-o", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")

    # model
    parser.add_argument("--model_name", "-m", type=str, default="google/deplot")
    parser.add_argument("--max_enc_length", type=int, default=128)
    parser.add_argument("--max_dec_length", type=int, default=512)
    parser.add_argument("--max_patches", type=int, default=2048)

    # training
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=2)
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
    parser.add_argument("--eval_batch_size", type=int, default=8)
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
    eval_result = main(args, seed)
