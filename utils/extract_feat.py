import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)

from models import OpenCLIP_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int)
    return parser.parse_args()


def process_line(line):
    path, label = line.split()
    path = f"data/imagenet/val/{path}"
    pil_image = Image.open(path).convert("RGB")
    array = np.array(pil_image) 
    return int(label), array


def to_tensor(array):
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


if __name__ == "__main__":
    model_id = get_args().model_id

    imagenet_path = "data/imagenet/meta/val.txt"
    imagenet_lookup = {}
    with open(imagenet_path, "r") as f:
        lines = f.readlines()

    print("Loading Data...")
    with mp.Pool(32) as pool:
        results = list(tqdm(pool.imap(process_line, lines), total=len(lines)))

    for label, array in results:
        tensor = to_tensor(array)
        if label in imagenet_lookup:
            imagenet_lookup[label].append(tensor)
        else:
            imagenet_lookup[label] = [tensor]

    print("Data Loaded!")

    model = OpenCLIP_model(model_id, "cuda")

    all_feat = []

    for class_id in tqdm(range(1000)):
        feat = model.get_image_feature(imagenet_lookup[class_id])
        all_feat.append(feat.cpu())

    all_feat = torch.stack(all_feat).numpy()
    os.makedirs("data/feat", exist_ok=True)
    np.save(f"data/feat/openclip_{model_id}_imagenet_feat.npy", all_feat)

