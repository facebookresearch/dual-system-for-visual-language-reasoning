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
            raw_prediction = generation_span[1].replace('.', '').strip()
            prediction = ord(raw_prediction[1]) - ord('a')
    
        if "answer" in annotation:
            answer = annotation["answer"]
        elif "correct_choice_idx" in annotation:
            answer = annotation["correct_choice_idx"]
        else:
            print("Not implemented!")
            return
        debug_flag = 0
        if prediction == answer:
            accuracy += 1
            debug_flag = 1
    
        debug_file.write(json.dumps({"answer": answer, "prediction": raw_prediction, "correct": debug_flag}) + "\n")

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
