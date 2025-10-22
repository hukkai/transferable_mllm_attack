from typing import Tuple

import open_clip
import torch
import torch.nn.functional as F
import torchvision

from .oclip_forward import forward as oclip_forward


supported_models = [
    ("ViT-H-14-378-quickgelu", "dfn5b"),
    ("ViT-H-14-quickgelu", "dfn5b"),
    ("ViT-SO400M-14-SigLIP-384", "webli"),
    ("ViT-SO400M-14-SigLIP", "webli"),
    ("ViT-L-16-SigLIP-384", "webli"),
    ("ViT-bigG-14", "laion2b_s39b_b160k"),
    ("ViT-H-14-CLIPA-336", "datacomp1b"),
    ("ViT-H-14-quickgelu", "metaclip_fullcc"),
    ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"),
]


class OpenCLIP_transform(object):

    def __init__(self, image_processor):
        super(OpenCLIP_transform, self).__init__()
        self.transforms = image_processor.transforms

    def __call__(self, image):
        for transform in self.transforms:
            if isinstance(transform, torch.nn.Module):
                image = transform(image)
            elif isinstance(transform, torchvision.transforms.ToTensor):
                image = image.float().clamp(0, 255) / 255
        return image


class OpenCLIP_model(object):

    def __init__(
        self,
        model_id: int,
        device: str = "cpu",
        force_patch_dropout: float = 0.0,
        force_drop_path: float = 0.0,
        model_dtype: str = "float32",
    ):
        super(OpenCLIP_model, self).__init__()
        model_name, pretrained = supported_models[model_id]

        self.model_id = model_id
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device

        input_kwargs = dict()

        if "ViT" in model_name:
            input_kwargs["force_patch_dropout"] = force_patch_dropout

        model_config = open_clip.get_model_config(model_name)
        if model_config is None or "vision_cfg" not in model_config:
            raise ValueError(
                f"Model config for '{model_name}' is missing or does not contain 'vision_cfg'"
            )

        vision_cfg = model_config["vision_cfg"]
        self.input_size = vision_cfg["image_size"]
        if "timm_model_name" in vision_cfg:
            vision_cfg["timm_drop_path"] = force_drop_path
            input_kwargs["vision_cfg"] = vision_cfg

        model, preprocess = open_clip.create_model_from_pretrained(
            model_name=model_name, pretrained=pretrained, **input_kwargs
        )  # pyright: ignore

        model.requires_grad_(False)
        self.model = model

        if force_patch_dropout > 0 or force_drop_path > 0:
            self.model.train()
        else:
            self.model.eval()

        if model_dtype == "float16":
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32


        self.model.to(device=device, dtype=self.model_dtype)

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.image_transform = OpenCLIP_transform(preprocess)

    def get_prompt(self, text_pair):
        text_inputs = self.tokenizer(text_pair).to(self.device)
        return text_inputs

    def get_image_feature(self, images, drop_rate=0.0, out_hidden_idx=0):
        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        for image in images:
            image = image.to(self.device)
            pixel_value = self.image_transform(image)
            pixel_values.append(pixel_value)

        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(self.model_dtype)

        # image_feats = self.model.encode_image(pixel_values)
        image_feats = oclip_forward(
            self.model, pixel_values, drop_rate=drop_rate, out_hidden_idx=out_hidden_idx
        )
        image_feats = image_feats.float()
        image_feats = F.normalize(image_feats, dim=1)
        return image_feats

    def get_text_feature(self, texts):
        text_ids = self.tokenizer(texts).to(self.device)
        text_feats = self.model.encode_text(text_ids)  # pyright: ignore
        text_feats = text_feats.float()
        text_feats = F.normalize(text_feats, dim=1)
        return text_feats

    def compute_loss(self, images, target_text, untarget_text, temperature=0.1):
        image_feat = self.get_image_feature(images)

        if isinstance(target_text, str):
            target_text = [target_text]
        if isinstance(untarget_text, str):
            untarget_text = [untarget_text]

        num_targets = len(target_text)
        text_pair = target_text + untarget_text
        text_feats = self.get_text_feature(text_pair)

        logits = image_feat @ text_feats.T
        probs = logits.div(temperature).softmax(dim=-1)

        loss = -probs[:, :num_targets].log().mean()
        return loss

    def __repr__(self):
        extra = f"model={self.model_id}, pretrained={self.pretrained}"
        return f"{self.__class__.__name__}({extra})"
