import numpy as np
import torch
from PIL import Image


def to_tensor(image, input_size=None):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if type(input_size) is int:
        image = image.resize((input_size, input_size))

    array = np.array(image)
    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    return tensor.float()


def to_pil(tensor):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    array = tensor.to(torch.uint8).numpy()
    image = Image.fromarray(array)
    return image


def get_random_crop(H, W):
    count = 10
    while count > 0:
        area = H * W * torch.Tensor(1).uniform_(0.8**2, 1).item()
        ratio = torch.Tensor(1).uniform_(9 / 10, 10 / 9).item()
        ch = round(area**0.5 * ratio**0.5)
        cw = round(area**0.5 / ratio**0.5)
        if ch <= H and cw <= W:
            h = torch.randint(H - ch + 1, size=[]).item()
            w = torch.randint(W - cw + 1, size=[]).item()
            return slice(h, h + ch), slice(w, w + cw)
        count -= 1
    ch = cw = min(H, W)
    h = torch.randint(H - ch + 1, size=[]).item()
    w = torch.randint(W - cw + 1, size=[]).item()
    return slice(h, h + ch), slice(w, w + cw)
