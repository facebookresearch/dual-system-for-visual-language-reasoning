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
from PIL import Image

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Tokenizer, LLaMA
from llama.model_with_past import Transformer
from transformers import set_seed, AutoProcessor, Pix2StructForConditionalGeneration

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
    print("Vocab size:", tokenizer.n_words)
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
        prompt = importlib.import_module("prompts.chartQA.{}".format(args.prompt))
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

    # ----------------------------------------------------- #
    # load llama
    args.local_rank, args.world_size = setup_model_parallel()
    args.device = torch.device("cuda:{}".format(args.local_rank))
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
    # load pix2struct
    processor = AutoProcessor.from_pretrained(
        args.vlqa_name, cache_dir=os.path.join(args.home_dir, "hg_cache")
    )
    processor.image_processor.is_vqa = False
    model = Pix2StructForConditionalGeneration.from_pretrained(args.vlqa_dir)
    model.eval()
    model.to(args.device)

    # ----------------------------------------------------- #
    # inference

    num_batch = -(args.num_beams // -args.eval_batch_size)
    output_path = os.path.join(
        args.output_prefix,
        "inference_split{}-{}.jsonl".format(args.split, args.num_process),
    )
    if os.path.exists(output_path) and (not args.overwrite_prediction):
        with open(output_path, "r") as fr:
            restart_line = len(fr.readlines())
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

        image = Image.open(os.path.join(args.image_dir, image_name + ".png"))

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

        input_seqs = [
            prompt.PROMPT.format(question=question) for beam_id in range(args.num_beams)
        ]

        generations = ["" for beam_id in range(args.num_beams)]
        inferences = ["" for beam_id in range(args.num_beams)]
        beam_finish = [False for beam_id in range(args.num_beams)]
        cnt_query = 0
        batch_past_key_values = [None for batch_id in range(num_batch)]
        batch_prev_pos = [0 for batch_id in range(num_batch)]
        while True:
            if args.debug or eid < 5:
                print(input_seqs[0])
            all_batch_inferences = []
            for batch_id in range(num_batch):
                batch_input_seqs = input_seqs[
                    batch_id
                    * args.eval_batch_size : min(
                        (batch_id + 1) * args.eval_batch_size, args.num_beams
                    )
                ]
                past_key_values = batch_past_key_values[batch_id]
                prev_pos = batch_prev_pos[batch_id]
                try:
                    (
                        batch_inferences,
                        past_key_values,
                        prev_pos,
                    ) = generator.generate_with_past(
                        batch_input_seqs,
                        stop_tokens="\n",
                        max_gen_len=args.max_gen_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        past_key_values=past_key_values,
                        prev_pos=prev_pos,
                    )
                except Exception as e:
                    print(e)
                    batch_inferences = ["" for beam_id in range(len(batch_input_seqs))]
                all_batch_inferences.extend(batch_inferences)
                batch_past_key_values[batch_id] = past_key_values
                batch_prev_pos[batch_id] = prev_pos

            for beam_id in range(args.num_beams):
                if not beam_finish[beam_id]:
                    generations[beam_id] += all_batch_inferences[beam_id].strip() + "\n"
                    if cnt_query == 0:
                        input_seqs[beam_id] += " "
                    input_seqs[beam_id] += all_batch_inferences[beam_id].strip() + "\n"

                    if "answer" in all_batch_inferences[beam_id]:
                        beam_finish[beam_id] = True
                        inferences[beam_id] = all_batch_inferences[beam_id].strip()

            if args.debug or eid < 5:
                print(input_seqs[0])

            if all(beam_finish):
                break

            for beam_id in range(args.num_beams):
                if beam_finish[beam_id]:
                    continue
                inference = all_batch_inferences[beam_id]
                # if "Let's" in inference:
                inference = "Q: " + inference.strip() + " A:"
                vl_inputs = processor(
                    images=image,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_patches=args.max_patches,
                ).to(args.device)
                question_ids = processor.tokenizer(
                    text=[inference], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(args.device)
                input_length = len(question_ids[0])
                vl_output = model.generate(
                    **vl_inputs,
                    decoder_input_ids=question_ids,
                    max_new_tokens=args.max_vlqa_len,
                )[0]
                inference = processor.decode(
                    vl_output[input_length + 1 :], skip_special_tokens=True
                )
                if "The x-axis shows: " in inference:
                    xaxis = inference.split("The x-axis shows: ")[1]
                    new_inference = (
                        inference.split("The x-axis shows: ")[0] + "The x-axis shows: "
                    )
                    visited = set()
                    nxaxis = []
                    for xi, x in enumerate(xaxis.split(" | ")):
                        x = x.strip()
                        if xi == len(xaxis.split(" | ")) - 1 and x.endswith("."):
                            x = x[:-1]
                        if x in visited:
                            continue
                        visited.add(x)
                        nxaxis.append(x)
                    new_inference += " | ".join(nxaxis) + "."
                    inference = new_inference
                generations[beam_id] += inference.strip() + "\n"

                input_seqs[beam_id] += inference.strip() + "\n"
                if "answer" in inference:
                    beam_finish[beam_id] = True
                    inferences[beam_id] = inference.strip()

            if all(beam_finish) or cnt_query > 9:
                break
            cnt_query += 1
        output = example.copy()
        output["meta"] = generations
        output["inference"] = inferences
        if args.local_rank == 0:
            fw.write(json.dumps(output) + "\n")
    fw.close()

    # ----------------------------------------------------- #


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run main.")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--home_dir", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--output_prefix", "-o", type=str)
    parser.add_argument("--prompt", "-p", type=str)
    parser.add_argument("--eval_split", type=str, default="test,dev,train")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite_prediction", action="store_true", default=False)

    # vlqa
    parser.add_argument("--vlqa_name", type=str)
    parser.add_argument("--vlqa_mode", type=str)
    parser.add_argument("--vlqa_dir", type=str)
    parser.add_argument("--max_patches", type=int, default=2048)
    parser.add_argument("--max_vlqa_len", type=int, default=128)

    # decoding strategy
    parser.add_argument("--instruct", type=str, default=None)
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
