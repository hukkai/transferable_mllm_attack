import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from diff_jpeg import DiffJPEGCoding

from models import get_random_crop, OpenCLIP_model, to_pil, to_tensor
from tqdm import tqdm

from utils.setup_env import initialize_DDP


diff_jpeg_coding_module = DiffJPEGCoding()


def to_jpeg(x):
    B, _, H, W = x.shape
    h = H // 16 * 16
    w = W // 16 * 16
    _h = torch.randint(H - h + 1, size=[1]).item()
    _w = torch.randint(W - w + 1, size=[1]).item()
    x = x[..., _h : _h + h, _w : _w + w]
    jpeg_quality = torch.rand(B).to(x.device) * 49 + 50
    return diff_jpeg_coding_module(x, jpeg_quality=jpeg_quality)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launcher",
        type=str,
        default="pytorch",
        help="should be either `slurm` or `pytorch`",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--num_eg", type=int, default=50)
    parser.add_argument("--input_size", type=int, default=299)
    parser.add_argument("--eps", type=int, default=8)
    parser.add_argument("--exclude", type=int, default=8)
    parser.add_argument("--model_dtype", type=str, default="float16")

    parser.add_argument("--batch_size", type=int, default=25)

    return parser.parse_args()


def read_csv(path):
    with open(path) as f:
        contents = csv.reader(f, delimiter=",")
        contents = [i for i in contents]
    return contents[1:]


def read_dataset(path="./data/nips2017_adv_dev/"):
    categories = read_csv(f"{path}/categories.csv")
    lookup = {}
    for category_id, category_text in categories:
        lookup[category_id] = category_text

    image_list = read_csv(f"{path}/images.csv")
    output = []
    for i in image_list:
        data = SimpleNamespace()
        data.image_id = i[0]
        data.image_path = f"{path}/images/{i[0]}.png"
        data.gt = int(i[6])
        data.target = int(i[7])
        data.gt_text = lookup[i[6]]
        data.target_text = lookup[i[7]]

        output.append(data)

    return output


def random_pad(x, size):
    _, _, H, W = x.shape
    if W < size:
        left = int(torch.randint(size - W + 1, size=[1]).item())
        right = size - W - left
    else:
        left = right = 0

    if H < size:
        upper = int(torch.randint(size - H + 1, size=[]).item())
        lower = size - H - upper
    else:
        upper = lower = 0

    if torch.randn(1).item() > 0:
        value = 255
    else:
        value = 0

    x = torch.nn.functional.pad(x, pad=(left, right, upper, lower), value=value)
    return x


def main():

    args = get_args()
    N_eg = args.num_eg
    EPS = args.eps

    rank, _, world_size = initialize_DDP(args.launcher)

    data_root = f"results/s{args.input_size}_x{args.exclude}_eps{EPS}"


    os.makedirs(data_root, exist_ok=True)

    save_path = data_root + "/{i}.png"
    ema_path = data_root + "/ema_{i}.png"

    dataset = read_dataset()

    dataset = [j for i, j in enumerate(dataset) if i % world_size == rank]
    dataset = [i for i in dataset if not os.path.isfile(save_path.format(i=i.image_id))]
    len_data = len(dataset)

    model_list = [0, 1, 2, 3, 4, 5, 6, 7]
    models = [
        OpenCLIP_model(
            model_id=i,
            device="cuda",
            force_drop_path=0.1,
            force_patch_dropout=0.2,
            model_dtype=args.model_dtype,
        )
        for i in tqdm(model_list)
        if i != args.exclude
    ]

    feat_bank = [
        torch.from_numpy(np.load(f"data/feat/openclip_{i}_imagenet_feat.npy")).cuda()
        for i in model_list
        if i != args.exclude
    ]

    print("All weights downloaded!")

    batch_size = args.batch_size

    start_time = time.time()
    while len(dataset) > 0:
        batch, dataset = dataset[:batch_size], dataset[batch_size:]

        images = [to_tensor(data.image_path, args.input_size) for data in batch]
        images = torch.cat(images, dim=0).to(device="cuda", dtype=torch.float32)

        _, _, H, W = images.shape

        adv = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([adv], lr=len(batch) * 0.1)

        gt = [data.gt - 1 for data in batch]
        target = [data.target - 1 for data in batch]
        ema_adv = torch.zeros_like(adv)
        for step in range(1000):

            accumulate_grad = torch.zeros_like(adv)
            for model, feat in zip(models, feat_bank):
                optimizer.zero_grad()
                input_size = model.input_size

                adv_image = images + adv
                adv_image = adv_image + torch.randn_like(adv_image) * EPS / 4

                adv_image = adv_image.clamp(0, 255)

                if torch.randn(1).item() > 0:
                    slice_h, slice_w = get_random_crop(H, W)
                    adv_image = adv_image[..., slice_h, slice_w]

                if adv_image.shape[2] < input_size or adv_image.shape[3] < input_size:
                    if torch.randn(1).item() > 0:
                        adv_image = random_pad(adv_image, input_size)

                if torch.randn(1).item() > 0:
                    adv_image = to_jpeg(adv_image)

                adv_feat = model.get_image_feature(adv_image, drop_rate=0.1)  # 4, D

                batch_feat = torch.cat([feat[target, :N_eg], feat[gt, :N_eg]], dim=1)
                similarity = torch.einsum("bnd, bd -> bn", batch_feat.float(), adv_feat)
                log_probs = -similarity.log_softmax(dim=1)
                log_prob_target = log_probs[:, :N_eg]
                loss = log_prob_target.topk(
                    k=min(10, N_eg), largest=False, dim=1
                ).values.mean(dim=1)
                log_prob_gt = log_probs[:, N_eg : N_eg * 2].mean(dim=1)
                (loss - log_prob_gt).sum().backward()

                if adv.grad is None:
                    raise ValueError("adv.grad is not None")
                else:
                    accumulate_grad += adv.grad.float()

            adv.grad = accumulate_grad
            optimizer.step()
            _adv = adv.data.clamp(-EPS * 1.25, EPS * 1.25)
            _adv = (images + _adv).clamp(0, 255) - images
            adv.data.copy_(_adv)

            if step == 200:
                ema_adv = adv.data.clone()

            if step > 200:
                ema_adv += (adv.data.clone() - ema_adv) * 0.01

        adv.data.copy_(adv.data.round())
        save_images = (images + adv.clamp(-EPS, EPS)).round()
        for save_image, data in zip(save_images, batch):
            pil_img = to_pil(save_image)
            pil_img.save(save_path.format(i=data.image_id))

        save_images = (images + ema_adv.clamp(-EPS, EPS)).round()
        for save_image, data in zip(save_images, batch):
            pil_img = to_pil(save_image)
            pil_img.save(ema_path.format(i=data.image_id))

        speed = (time.time() - start_time) / (len_data - len(dataset))
        eta = speed * len(dataset) / 60
        print(f"{len(dataset)} unfinished, eta {eta: .1f} mins.")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()

