import json
import csv
import os
import numpy as np
import random
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse

def get_row_question(entity):
    if entity is None:
        q = "Let's extract all the values."
    else:
        q = "Let's extract the data of {}.".format(entity)
    return q

def get_row_answer(values, headers):
    if any(float(v) < 0 for v in values):
        return None
    a = "The data is {} in {}".format(values[0], headers[0])
    for v, h in zip(values[1:], headers[1:]):
        a += ", {} in {}".format(v, h)
    a += "."
    return a

def get_element_question(entity1, entity2=None):
    if entity2 is None:
        q = "Let's extract the data of {}.".format(entity1)
    else:
        q = "Let's extract the data of {} BY {}.".format(entity1, entity2)
    return q
def get_element_answer(value):
    if float(value) < 0:
        return None
    a = "The data is {}.".format(value)
    return a

def get_row_qa_by_row(example):
    sample_entity_id = random.choice(range(1, example['table'].shape[0]))
    sample_entity = example['table'][sample_entity_id][0]
    values = example['table'][sample_entity_id][1:]
    headers = example['table'][0][1:]
    q = get_row_question(sample_entity)
    a = get_row_answer(values, headers)
    return q, a
def get_row_qa_by_column(example):
    sample_entity_id = random.choice(range(1, example['table'].shape[1]))
    sample_entity = example['table'][0][sample_entity_id]
    values = example['table'][1:, sample_entity_id]
    headers = example['table'][1:, 0]
    q = get_row_question(sample_entity)
    a = get_row_answer(values, headers)
    return q, a
def get_row_qa_by_single_row(example):
    values = example['table'][1][1:]
    headers = example['table'][0][1:]
    q = get_row_question(example['table'][1][0])
    a = get_row_answer(values, headers)
    return q, a
def get_row_qa_by_single_column(example):
    values = example['table'][1:, 1]
    headers = example['table'][1:, 0]
    q = get_row_question(example['table'][0][1])
    a = get_row_answer(values, headers)
    return q, a

def get_element_qa_by_row(example):
    sample_entity_id = random.choice(range(1, example['table'].shape[0]))
    sample_entity = example['table'][sample_entity_id][0]
    sample_entity_id2 = random.choice(range(1, example['table'].shape[1]))
    sample_entity2 = example['table'][0][sample_entity_id2]
    value = example['table'][sample_entity_id][sample_entity_id2]
    q = get_element_question(sample_entity, sample_entity2)
    a = get_element_answer(value)
    return q, a
def get_element_qa_by_column(example):
    sample_entity_id2 = random.choice(range(1, example['table'].shape[0]))
    sample_entity2 = example['table'][sample_entity_id2][0]
    sample_entity_id = random.choice(range(1, example['table'].shape[1]))
    sample_entity = example['table'][0][sample_entity_id]
    value = example['table'][sample_entity_id2][sample_entity_id]
    q = get_element_question(sample_entity, sample_entity2)
    a = get_element_answer(value)
    return q, a
def get_element_qa_by_row_single(example):
    sample_entity_id = random.choice(range(1, example['table'].shape[0]))
    sample_entity = example['table'][sample_entity_id][0]
    value = example['table'][sample_entity_id][1]
    q = get_element_question(sample_entity)
    a = get_element_answer(value)
    return q, a
def get_element_qa_by_column_single(example):
    sample_entity_id = random.choice(range(1, example['table'].shape[1]))
    sample_entity = example['table'][0][sample_entity_id]
    value = example['table'][1][sample_entity_id]
    q = get_element_question(sample_entity)
    a = get_element_answer(value)
    return q, a

def clean_table(example):
    new_table = []
    for row in example["table"]:
        new_table.append([v.replace('*', '') for v in row])
    example["table"] = np.asarray(new_table)
    return example

def get_number(value):
    values = re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', value)
    if len(values) == 0:
        return "nan"
    try:
        value = float(values[0].replace(',', ''))
    except:
        return "nan"
    return value

def load_tables(tables_dir, source):
    file_names = os.listdir(tables_dir)
    tables = []
    for file in tqdm(file_names):
        if not file.endswith('.csv'):
            continue
        file_path = os.path.join(tables_dir, file)
        with open(file_path, 'r') as fr:
            csvFile = csv.reader(fr)
            table = []
            line_id = 0
            for line in csvFile:
                line_id += 1
                if line_id == 1 and source == "plotQA":
                    continue
                if line_id > 2:
                    line = [line[0]] + [get_number(v) for v in line[1:]]
                table.append(line)
        tables.append({"id": file.replace('.csv', ''), "table": np.asarray(table)})
    return tables

def main(args):
    tables = load_tables(args.tables_dir, args.source)
    with open(args.output_path, 'w') as fw:
        for example in tables:
            example = clean_table(example)
            # get row qa
            qset = set()
            for sample_idx in range(5):
                q, a = None, None
                try:
                    if example['table'].shape[0] > 2 and example['table'].shape[1] > 2:
                        if random.random() < 0.5:
                            q, a = get_row_qa_by_column(example)
                        else:
                            q, a = get_row_qa_by_row(example)
                    elif example['table'].shape[0] > 2 and example['table'].shape[1] == 2:
                        q, a = get_row_qa_by_single_column(example)
                    elif example['table'].shape[0] == 2 and example['table'].shape[1] > 2:
                        q, a = get_row_qa_by_single_row(example)
                except:
                    q, a = None, None
                if q is None or a is None:
                    continue
                if q in qset:
                    continue
                if 'nan' in a or 'nan' in q:
                    continue
                if '-.' in a or '- in' in a:
                    continue
                qset.add(q)
                output = {"id": example["id"], "type": "row/column", "question": q, "answer": a, "source": args.source}
                fw.write(json.dumps(output)+"\n")
            qset = set()
            for sample_idx in range(5):
                q, a = None, None
                try:
                    if example['table'].shape[0] > 2 and example['table'].shape[1] > 2:
                        if random.random() < 0.5:
                            q, a = get_element_qa_by_column(example)
                        else:
                            q, a = get_element_qa_by_row(example)
                    elif example['table'].shape[0] > 2 and example['table'].shape[1] == 2:
                        q, a = get_element_qa_by_row_single(example)
                    elif example['table'].shape[0] == 2 and example['table'].shape[1] > 2:
                        q, a = get_element_qa_by_column_single(example)
                except:
                    q, a = None, None
                if q is None or a is None:
                    continue
                if q in qset:
                    continue
                if 'nan' in a or 'nan' in q:
                    continue
                if '-.' in a or '- in' in a:
                    continue
                qset.add(q)
                output = {"id": example["id"], "type": "element", "question": q, "answer": a, "source": args.source}
                fw.write(json.dumps(output)+"\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--source', type=str)
    parser.add_argument('--tables_dir', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    main(args)