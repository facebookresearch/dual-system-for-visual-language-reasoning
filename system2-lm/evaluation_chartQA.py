import os
import sys
import json

import argparse
from tqdm import tqdm
from collections import defaultdict

def get_float(string):
    try:
        # if string.endswith("%"):
        #     return float(string.rstrip("%")) / 100.0
        # else:
        # return abs(float(string))
        return float(string)
            # return float("{:.2f}".format(float(string)))
    except ValueError:
        return None

def parse_prediction(generation):
    prediction = ""
    generation_span = generation.split("Q:")[0].split("he answer is ")
    if len(generation_span) > 1:
        if ". " in generation_span[-1]:
            generation_span = generation_span[-1].split('. ')[0]
        else:
            generation_span = generation_span[-1]
        prediction = generation_span.strip().replace('*', '').replace('%', '')
        if prediction.endswith('.'):
            prediction = prediction[:-1]
    return prediction

def majority_vote(List):
    return max(set(List), key = List.count)

def main(args):
    # load predictions
    with open(args.prediction_file, 'r') as fr:
        duplicated_predictions = [json.loads(line) for line in fr.readlines()]
    question_ids = set()
    raw_predictions = []
    for pred in duplicated_predictions:
        # qid = str(pred["image_index"]) + str(pred["qid"]) + pred["question_string"] # + str(pred["answer_id"]) + str(pred["question_id"])
        # if not qid in question_ids:
        #     raw_predictions.append(pred)
        #     question_ids.add(qid)
        raw_predictions.append(pred)
    print("Evaluating {} predictions...".format(len(raw_predictions)))

    debug_path = args.prediction_file.replace('.jsonl', '_debug.jsonl')
    debug_file = open(debug_path, 'w')
    
    relax_accuracy = 0.
    exact_accuracy = 0.
    exact_accuracy_by_type = defaultdict(list)
    for example in raw_predictions:
        if isinstance(example["inference"], list):
            # generation = example["inference"][-1]
            # prediction = parse_prediction(generation)
            prediction_list = []
            for generation in example["inference"]:
                prediction_list.append(parse_prediction(generation))
            
            prediction = majority_vote(prediction_list)
        else:
            generation = example["inference"]
            prediction = parse_prediction(generation)
       
        if "answer" in example:
            answer = str(example["answer"]).strip().replace('*', '').replace('%', '')
        else:
            answer = str(example["label"]).strip().replace('*', '').replace('%', '')

        # if prediction == answer:
        #     exact_accuracy += 1

        processed_prediction = get_float(prediction)
        processed_answer = get_float(answer)

        debug_flag = 0
        if processed_prediction is None or processed_answer is None:
            # Use exact match for textual answers
            # if prediction.lower() in ["true", "false"]:
            #     if prediction.lower() == "true":
            #         prediction = "yes"
            #     elif prediction.lower() == "false":
            #         prediction = "no"
            if prediction.lower() == answer.lower():
                exact_accuracy += 1
                relax_accuracy += 1
                debug_flag = 1

            if "image_index" in example:
                debug_file.write(json.dumps({"id": example["image_index"], "answer": answer, "prediction": prediction, "correct": debug_flag}) + "\n")
            elif "image" in example:
                debug_file.write(json.dumps({"id": example["image"], "answer": answer, "prediction": prediction, "correct": debug_flag}) + "\n")
            else:
                debug_file.write(json.dumps({"id": example["imgname"], "answer": answer, "prediction": prediction, "correct": debug_flag}) + "\n")
        else:
            if processed_prediction == processed_answer:
                exact_accuracy += 1
            # Use relax match for numbers
            if processed_answer != 0:
                if abs(processed_prediction - processed_answer) / abs(processed_answer) <= 0.05:
                    relax_accuracy += 1
                    debug_flag = 1
            else:
                if processed_prediction == processed_answer:
                    relax_accuracy += 1
                    debug_flag = 1
            # elif processed_answer != 0 and processed_answer < 1.0 and abs(processed_prediction - processed_answer * 100.) / (processed_answer * 100.) < 0.05:
            #     print(processed_answer, processed_prediction)
            #     relax_accuracy += 1
                # debug_flag = 1
                
            if "image_index" in example:
                debug_file.write(json.dumps({"id": example["image_index"], "answer": processed_answer, "prediction": processed_prediction, "correct": debug_flag}) + "\n")
            elif "image" in example:
                debug_file.write(json.dumps({"id": example["image"], "answer": answer, "prediction": prediction, "correct": debug_flag}) + "\n")
            else:
                debug_file.write(json.dumps({"id": example["imgname"], "answer": processed_answer, "prediction": processed_prediction, "correct": debug_flag}) + "\n")
        # exact_accuracy_by_type[example["template"]].append(debug_flag)
    
    debug_file.close()
    # for type in exact_accuracy_by_type:
    #     exact_accuracy_by_type[type] = sum(exact_accuracy_by_type[type]) * 100. / len(exact_accuracy_by_type[type])

    relax_accuracy /= len(raw_predictions)
    exact_accuracy /= len(raw_predictions)
    result_file = args.prediction_file.replace('.jsonl', '_evaluation.json')
    with open(result_file, 'w') as fw:
        json.dump({"exact_accuracy": exact_accuracy * 100., "relax_accuracy": relax_accuracy * 100., "accuracy_by_type": exact_accuracy_by_type}, fw, indent=4)
    print("Evaluation done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--prediction_file', '-p', type=str)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    main(args)
