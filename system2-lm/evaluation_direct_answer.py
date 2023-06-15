import os
import sys
import json

import argparse
from tqdm import tqdm

def main(args):
    # load annotations 
    with open(args.annotation_file, 'r') as fr:
        annotations = [json.loads(line) for line in fr.readlines()]

     # load predictions
    with open(args.prediction_file, 'r') as fr:
        raw_predictions = [json.loads(line) for line in fr.readlines()]
    print("Evaluating {} predictions...".format(len(raw_predictions)))

    debug_path = args.prediction_file.replace('.jsonl', '_debug.jsonl')
    debug_file = open(debug_path, 'w')
 
    accuracy = 0.
    for annotation, example in zip(annotations, raw_predictions):
        generation = example["inference"][0]
        prediction = ""
        raw_prediction = generation
        generation_span = generation.split("the answer is ")
        if len(generation_span) == 2:
            prediction = generation_span[1].replace('.', '').strip()
    
        if "direct_answers" in annotation:
            cnt_hit = len([human_answer for human_answer in annotation["direct_answers"] if human_answer == prediction])
            debug_flag = min(1, cnt_hit / 3.0)
            accuracy += debug_flag
        else:
            answer = annotation["choices"]["answer"]
    
        debug_file.write(json.dumps({"correct": debug_flag, "prediction": prediction, "answer": annotation["direct_answers"]}) + "\n")

    debug_file.close()
    accuracy /= len(raw_predictions)
    result_file = args.prediction_file.replace('.jsonl', '_evaluation.json')
    with open(result_file, 'w') as fw:
        json.dump({"Accuracy": accuracy * 100.}, fw, indent=4)
    print("Evaluation done!")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--annotation_file', '-a', type=str)
    parser.add_argument('--prediction_file', '-p', type=str)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    main(args)
