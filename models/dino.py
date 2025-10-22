import torch
import torch.nn.functional as F
from torchvision import transforms as tv_trans
from transformers import AutoImageProcessor, AutoModel


class transform(object):

    def __init__(self, image_processor, device):
        super(transform, self).__init__()

        # resize
        interpolation = getattr(
            image_processor, "resample", tv_trans.InterpolationMode.BICUBIC
        )
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        elif "height" in image_processor.size:
            size = (image_processor.size["height"], image_processor.size["width"])
        else:
            raise ValueError("Unsuported `image_processor`")

        self.resize = tv_trans.Resize(
            size=size, interpolation=interpolation, antialias=True
        )

        # center crop
        do_center_crop = getattr(image_processor, "do_center_crop", False)
        if do_center_crop and hasattr(image_processor, "crop_size"):
            crop_size = image_processor.crop_size
            ch = image_processor.crop_size["height"]
            cw = image_processor.crop_size["width"]
            self.resize = tv_trans.Compose([self.resize, tv_trans.CenterCrop((ch, cw))])

        self.rescale_factor = image_processor.rescale_factor

        mean = torch.tensor(image_processor.image_mean).to(device)
        std = torch.tensor(image_processor.image_std).to(device)

        self.mean = mean.view(3, 1, 1)
        self.std = std.view(3, 1, 1)

        # overwrite!
        self.resize = tv_trans.Resize(
            size=336, interpolation=interpolation, antialias=True
        )

    def __call__(self, image):
        image = self.resize(image)
        image = image.float().clamp(0, 255)
        image = image * self.rescale_factor
        image = (image - self.mean) / self.std
        return image


class dino_model(object):

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        model_dtype: str = "float32",
    ):
        super(dino_model, self).__init__()
        self.model_id = model_id
        self.device = device

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

        if model_dtype == "bfloat16":
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32

        self.model.to(device=device, dtype=self.model_dtype)

        self.image_transform = transform(processor, device)
        self.input_size = 336

    def get_image_feature(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        for image in images:
            image = image.to(self.device)
            pixel_value = self.image_transform(image)
            pixel_values.append(pixel_value)

        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(self.model_dtype)

        outputs = self.model(pixel_values=pixel_values)
        image_feats = outputs.pooler_output
        image_feats = F.normalize(image_feats, dim=1)
        return image_feats

    def __repr__(self):
        extra = f"model={self.model_id}"
        return f"{self.__class__.__name__}({extra})"
