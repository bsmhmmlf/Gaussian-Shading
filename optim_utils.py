import torch
from datasets import load_dataset
from typing import Any, Mapping
import json
import numpy as np
import os
from statistics import mean, stdev


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def get_dataset(args):
    if 'laion' in args.dataset_path:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset_path:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset_path)['train']
        prompt_key = 'Prompt'
    return dataset, prompt_key


def save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores):
    names = {
        'jpeg_ratio': "Jpeg.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'random_drop_ratio': "RandomDrop.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'median_blur_k': "MedBlur.txt",
        'resize_ratio': "Resize.txt",
        'sp_prob': "SPNoise.txt",
        'brightness_factor': "Color_Jitter.txt"
    }
    filename = "Identity.txt"
    for option, name in names.items():
        if getattr(args, option) is not None:
            filename = name

    if args.reference_model is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       'mean_clip_score:' + str(mean(clip_scores)) + '      ' + 'std_clip_score:' + str(stdev(clip_scores)) + '      ' +
                       '\n')

    else:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       '\n')

    return





