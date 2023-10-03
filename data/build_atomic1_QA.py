import json
import csv
import os
import numpy as np
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse

import webcolors
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_color_name2(rgb):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "light blue": (40, 118, 221),
        "dark blue": (15, 40, 62),
        "yellow": (255, 255, 0),
        "pink": (255, 0, 255),
        "light green": (0, 255, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0),
        "brown": (165, 42, 42),
    }
    min_distance = float("inf")
    closest_color = None
    for color, value in colors.items():
        distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def get_color_name1(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return closest_name

def get_color(_hex):
    try:
        color1 = get_color_name1(tuple(webcolors.hex_to_rgb(_hex)))
        color2 = get_color_name2(tuple(webcolors.hex_to_rgb(_hex)))
        if color1 == "silver":
            return "grey"
        if "green" in color1 and not "green" in color2:
            if "dark" in color1:
                return "dark green"
            if "light" in color1:
                return "light green"
            return "green"
#         if "blue" in color1 and not "blue" in color2:
#             return "blue"
        if color1 == "slategray":
            return "grey"
        if color1 == "mediumaquamarine":
            return "green"
        if color1 == "tan":
            return "brown"
        if color1 == "darkred":
            return "dard red"
        if color1 == "crimson":
            return "red"
        if color1 == "violet":
            return "pink"
        if color1 == "slateblue" or color1 == "mediumvioletred":
            return "purple"
        if color1 == "darkcyan" or color1 == "darkseagreen":
            return "green"
        if color1 == "teal":
            return "dark green"
        if color1 == "darkturquoise":
            return "light green"
        if color1 == "khaki" or color1 == "darkgoldenrod":
            return "brown"
        if color1 == "royalblue":
            return "blue"
        return color2
    except:
        return _hex

def build_qa_for_plotQA(input_path, output_path):
    with open(input_path, 'r') as fr:
        annotations = json.load(fr)
    # 'vbar_categorical', 'line', 'hbar_categorical', 'dot_line'
    fw = open(output_path, 'w')
    q = "Let's describe the figure."
    for annotation in tqdm(annotations):
        key = 'label' if 'label' in annotation["models"][0] else 'labels'
        row0 = [group[key] for group in annotation["models"]]
        key = 'color' if 'color' in annotation["models"][0] else 'colors'
        colors = [get_color(group[key]) for group in annotation["models"]]
        output_seq = "The figure shows the data of:"
        has_unk_color = False
        for hid, h in enumerate(row0):
            output_seq += " {}".format(h)
            if colors[hid] != "UNK" and colors[hid] != "":
                output_seq += ' ({})'.format(colors[hid])
            else:
                has_unk_color = True
            if hid < len(row0) - 1:
                output_seq += " |"
            else:
                output_seq += "."
        if len(colors) != len(set(colors)):
            continue
        if has_unk_color and len(row0) > 1:
            continue
        xaxis_key = 'x_axis'
        if annotation["type"] == 'hbar_categorical':
            xaxis_key = 'y_axis'
        output_seq += " The x-axis shows: {}.".format(' | '.join(annotation["general_figure_info"][xaxis_key]["major_labels"]["values"]))

        output = {"id": annotation["image_index"], "type": "desc", "question": q, "answer": output_seq, "source": 'plotQA'}
        fw.write(json.dumps(output)+"\n")
    fw.close()

def build_qa_for_chartQA(input_path, output_path):
    file_names = os.listdir(input_path)
    annotations = []
    for file in file_names:
        if not file.endswith('.json'):
            continue
        with open(os.path.join(input_path, file), 'r') as fr:
            example = json.load(fr)
            example["image_index"] = file.replace('.json', '')
            annotations.append(example)

    # {'line', 'h_bar', 'pie', 'v_bar'}
    fw = open(output_path, 'w')
    q = "Let's describe the figure."
    for annotation in tqdm(annotations):
        if len(annotation["models"]) == 0:
            continue
        if len(annotation["models"]) == 1 or annotation["type"] == "pie":
            if not 'general_figure_info' in annotation:
                header_name = "Value"
            elif 'title' in annotation['general_figure_info']:
                header_name = annotation['general_figure_info']['title']['text'].split(',')[0]
            elif 'y_axis' in annotation['general_figure_info']:
                header_name = annotation['general_figure_info']['y_axis']['label']['text'].split(',')[0]
            else:
                header_name = "Value"
            if len(header_name.split(' ')) >= 10:
                header_name = " ".join(header_name.split(' ')[:10])
            row0 = [header_name]
            if 'color' in annotation["models"][0]:
                colors = [get_color(annotation["models"][0]['color'])]
            elif 'colors' in annotation["models"][0]:
                colors = [get_color(annotation["models"][0]['colors'][0])]
            else:
                colors = ["UNK"]
        else:
            if "text_label" in annotation["models"][0]:
                key = "text_label" 
            elif "name" in annotation["models"][0]:
                key = "name"
            else:
                continue
            row0 = [a[key] for a in annotation["models"]]
            if "color" in annotation["models"][0]:
                colors = [get_color(a["color"]) for a in annotation["models"]]
            else:
                colors = [get_color(a["colors"][0]) if len(a["colors"]) > 0 else "UNK" for a in annotation["models"]]

        output_seq = "The figure shows the data of:"
        has_unk_color = False
        for hid, h in enumerate(row0):
            output_seq += " {}".format(h)
            if colors[hid] != "UNK" and colors[hid] != "":
                output_seq += ' ({})'.format(colors[hid])
            else:
                has_unk_color = True
            if hid < len(row0) - 1:
                output_seq += " |"
            else:
                output_seq += "."
        if has_unk_color and len(row0) > 1:
            continue
        if annotation["type"] != "pie" and "x_axis" in annotation["general_figure_info"]:
            
            xaxis_key = 'x_axis'

            xticks = [str(x) for x in annotation["general_figure_info"][xaxis_key]["major_labels"]["values"]]
            output_seq += " The x-axis shows: {}.".format(' | '.join(xticks))

        elif annotation["type"] == "pie":
            if "text_label" in annotation["models"][0]:
                key = "text_label" 
            elif "name" in annotation["models"][0]:
                key = "name"
            else:
                continue
            row0 = [str(a[key]) for a in annotation["models"]]
            output_seq += " The x-axis shows: {}.".format(' | '.join(row0))
            
        output = {"id": annotation["image_index"], "type": "header", "question": q, "answer": output_seq, "source": 'chartQA'}
        fw.write(json.dumps(output)+"\n")
    fw.close()

def main(args):
    if args.source == "chartQA":
        build_qa_for_chartQA(args.input_path, args.output_path)
    elif args.source == "plotQA":
        build_qa_for_plotQA(args.input_path, args.output_path)
    else:
        print("Not implemented!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--source', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    main(args)