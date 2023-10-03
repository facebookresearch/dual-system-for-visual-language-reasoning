# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import time
import json
import random
import numpy as np
import importlib

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Tokenizer, LLaMA
from llama.model_with_past import Transformer

import argparse
from tqdm import tqdm

# for REPRODUCIBILITY
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")  # , init_method='env://')
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    model_name: str,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_batch_size=max_batch_size, max_seq_len=max_seq_len, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(args):
    # ----------------------------------------------------- #
    # decide prompt
    try:
        prompt = importlib.import_module("prompts.tableQA.{}".format(args.prompt))
    except:
        print("No such prompt!")
        return
    if args.local_rank == 0:
        print(prompt.PROMPT)

    # ----------------------------------------------------- #
    with open(args.data_path, "r") as fr:
        all_lines = [json.loads(line) for line in fr.readlines()]

    batch_size = -(len(all_lines) // -args.num_process)
    testset = all_lines[(args.split * batch_size) : (args.split + 1) * batch_size]

    # load tables
    with open(args.table_path, "r") as fr:
        tables = [json.loads(line) for line in fr.readlines()]
    eid2table = {}
    for example in tables:
        eid2table[example["id"]] = example["table"]

    # ----------------------------------------------------- #
    # load model
    args.local_rank, args.world_size = setup_model_parallel()
    if args.local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    print(args.ckpt_dir)
    generator = load(
        args.ckpt_dir,
        args.tokenizer_path,
        args.local_rank,
        args.world_size,
        args.max_seq_len,
        args.eval_batch_size,
        args.model_name,
    )

    # ----------------------------------------------------- #
    # inference

    num_batch = -(args.num_beams // -args.eval_batch_size)
    output_path = os.path.join(
        args.output_prefix,
        "inference_split{}-{}.jsonl".format(args.split, args.num_process),
    )
    if os.path.exists(output_path) and (not args.overwrite_prediction):
        with open(output_path, "r") as fr:
            restart_line = int(len(fr.readlines()))
    else:
        restart_line = 0
    fw = open(output_path, "w" if args.overwrite_prediction else "a", buffering=1)
    for eid, example in enumerate(tqdm(testset[restart_line:])):
        if args.dataset == "chartQA":
            image_name = example["imgname"].replace(".png", "")
        elif args.dataset == "plotQA":
            image_name = str(example["image_index"])
        elif args.dataset == "dvQA":
            image_name = str(example["image"]).replace(".png", "")
        elif args.dataset == "figureQA":
            image_name = str(example["image_index"])
        else:
            print("Not implemented!")
            continue
        if not image_name in eid2table:
            print("No table for {}!".format(image_name))
            continue
        table = eid2table[image_name]
        formatted_table = prompt.formalize_table(table)

        if args.dataset == "chartQA":
            question = example["query"]
        elif args.dataset == "plotQA":
            question = example["question_string"]
        elif args.dataset == "dvQA":
            question = example["question"]
        elif args.dataset == "figureQA":
            question = example["question_string"]
        else:
            print("Not implemented!")
            continue
        input_seq = prompt.PROMPT.format(table=formatted_table, question=question)
        if args.debug or eid < 5:
            print(input_seq)

        generations = []
        for batch_id in range(num_batch):
            batch_inputs = [input_seq] * min(
                args.eval_batch_size, args.num_beams - batch_id * args.eval_batch_size
            )
            inference, _, _ = generator.generate_with_past(
                batch_inputs,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            generations.extend(inference)

        if args.local_rank == 0:
            output = example.copy()
            output["inference"] = [gen.strip() for gen in generations]
            fw.write(json.dumps(output) + "\n")
    fw.close()

    # ----------------------------------------------------- #


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main.")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--table_path", type=str)
    parser.add_argument("--output_prefix", "-o", type=str)
    parser.add_argument("--prompt", "-p", type=str)
    parser.add_argument("--eval_split", type=str, default="test,dev,train")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite_prediction", action="store_true", default=False)

    # decoding strategy
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--min_entropy", type=float, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # gpu and workers option
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int)

    args = parser.parse_args()

    main(args)
