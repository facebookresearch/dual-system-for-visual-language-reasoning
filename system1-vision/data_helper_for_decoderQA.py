import json
import os 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional
import random

import torch
from torch.utils.data import Dataset, TensorDataset
from PIL import Image

class ImageQA_Dataset(Dataset):

    def __init__(self, dataset, processor, max_patches=2048):
        self.dataset = dataset 
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], return_tensors="pt", add_special_tokens=True, max_patches=self.max_patches)
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["answer"] = item["answer"]
        encoding["question"] = item["question"]
        return encoding 

def load_raw_dataset(split, args):
    data_path = os.path.join(args.dataset, '{}.jsonl'.format(split))
    dataset = []

    with open(data_path, 'r') as fr:
        for line_idx, line in tqdm(enumerate(fr), desc='loading {}'.format(data_path)):
            example = json.loads(line)
            dataset.append(example)
    return dataset

def load_ImageQA_dataset(split, processor, args):
    data_path = os.path.join(args.dataset, '{}.jsonl'.format(split))
    dataset = []

    with open(data_path, 'r') as fr:
        for line_idx, line in tqdm(enumerate(fr), desc='processing {}'.format(data_path)):
            example = json.loads(line)
            image_path = os.path.join(args.image_dir, "{}.png".format(example["id"]))
            image = Image.open(image_path)
            dataset.append(
                    {
                        "image": image,
                        "question": example["question"],
                        "answer": example["answer"],
                        "type": example["type"]
                    }
            )
    # for example in dataset[:2]:
    #     print("*** Example ***")
    #     print(example)

    return ImageQA_Dataset(dataset, processor, args.max_patches)
    
class Data_Collator(object):
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args

    def __call__(self, batch):
        new_batch = {"flattened_patches":[], "attention_mask":[]}

        qa_texts = [self.processor.tokenizer.pad_token + item["question"] + " " + item["answer"] for item in batch]
        question_only_texts = [item["question"] for item in batch]
        qa_ids = self.processor.tokenizer(text=qa_texts, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True, max_length=self.args.max_dec_length).input_ids
        labels = qa_ids.new_zeros(qa_ids.shape)
        labels[..., :-1] = qa_ids[..., 1:].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for question_id, question in enumerate(question_only_texts):
            question_only_ids = self.processor.tokenizer(text=question, add_special_tokens=False).input_ids
            labels[question_id][:len(question_only_ids)] = -100
        
        new_batch["labels"] = labels 
        new_batch["decoder_input_ids"] = qa_ids 
        
        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
        
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

        return new_batch

class Data_Collator_for_inference(object):
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args

    def __call__(self, batch):
        new_batch = {"flattened_patches":[], "attention_mask":[]}

        qa_texts = [item["question"] for item in batch]
        qa_ids = self.processor.tokenizer(text=qa_texts, return_tensors="pt", add_special_tokens=False).input_ids
        
        new_batch["decoder_input_ids"] = qa_ids 
        
        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])
        
        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

        return new_batch