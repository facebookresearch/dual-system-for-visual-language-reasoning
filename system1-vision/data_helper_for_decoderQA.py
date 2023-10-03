# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageQA_Dataset(Dataset):
    def __init__(self, dataset, processor, max_patches=2048):
        self.dataset = dataset
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def load_raw_dataset(split, args):
    data_path = os.path.join("./data", args.dataset, "{}.jsonl".format(split))
    dataset = []

    with open(data_path, "r") as fr:
        for _, line in tqdm(enumerate(fr), desc="loading {}".format(data_path)):
            example = json.loads(line)
            dataset.append(example)
    return dataset


def load_ImageQA_dataset(split, processor, args):
    data_path = os.path.join("./data", args.dataset, "{}.jsonl".format(split))
    dataset = []

    with open(data_path, "r") as fr:
        for _, line in tqdm(enumerate(fr), desc="processing {}".format(data_path)):
            example = json.loads(line)
            dataset.append(
                {
                    "image": example["id"],
                    "question": example["question"],
                    "answer": example["answer"],
                    "source": example["source"],
                }
            )

    return ImageQA_Dataset(dataset, processor, args.max_patches)


class Data_Collator(object):
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args

    def __call__(self, batch):
        new_batch = {"flattened_patches": [], "attention_mask": []}
        for example in batch:
            image_path = os.path.join(
                self.args.image_dirs[example["source"]],
                "{}.png".format(example["image"]),
            )
            image = Image.open(image_path)
            encoding = self.processor(
                images=image,
                return_tensors="pt",
                add_special_tokens=True,
                max_patches=self.args.max_patches,
            )
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            new_batch["flattened_patches"].append(encoding["flattened_patches"])
            new_batch["attention_mask"].append(encoding["attention_mask"])

        qa_texts = [
            self.processor.tokenizer.pad_token
            + "Q: "
            + item["question"]
            + " A: "
            + item["answer"]
            for item in batch
        ]
        question_only_texts = ["Q: " + item["question"] + " A:" for item in batch]
        qa_ids = self.processor.tokenizer(
            text=qa_texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.args.max_dec_length,
        ).input_ids
        labels = qa_ids.new_zeros(qa_ids.shape)
        labels[..., :-1] = qa_ids[..., 1:].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for question_id, question in enumerate(question_only_texts):
            question_only_ids = self.processor.tokenizer(
                text=question, add_special_tokens=False
            ).input_ids
            labels[question_id][: len(question_only_ids)] = -100

        new_batch["labels"] = labels
        new_batch["decoder_input_ids"] = qa_ids

        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

        return new_batch


class Data_Collator_for_inference(object):
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args

    def __call__(self, batch):
        new_batch = {"flattened_patches": [], "attention_mask": []}
        for example in batch:
            image_path = os.path.join(
                self.args.image_dirs[example["source"]],
                "{}.png".format(example["image"]),
            )
            image = Image.open(image_path)
            encoding = self.processor(
                images=image,
                return_tensors="pt",
                add_special_tokens=True,
                max_patches=self.args.max_patches,
            )
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            new_batch["flattened_patches"].append(encoding["flattened_patches"])
            new_batch["attention_mask"].append(encoding["attention_mask"])

        qa_texts = ["Q: " + item["question"] + " A:" for item in batch]
        qa_ids = self.processor.tokenizer(
            text=qa_texts, return_tensors="pt", add_special_tokens=False
        ).input_ids

        new_batch["decoder_input_ids"] = qa_ids

        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

        return new_batch
