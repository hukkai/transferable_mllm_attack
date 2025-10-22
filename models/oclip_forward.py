import torch
from open_clip.transformer import VisionTransformer


def forward(model, x, drop_rate: float = 0, out_hidden_idx: int = 0):

    visual = model.visual
    if not isinstance(visual, VisionTransformer) or drop_rate == 0:
        return model.encode_image(x)

    x = visual._embeds(x)

    if not visual.transformer.batch_first:
        x = x.transpose(0, 1)  # NLD -> LND

    num_blocks = len(visual.transformer.resblocks)
    drop_rates = torch.linspace(0, drop_rate, num_blocks)
    drop_flags = (torch.rand([num_blocks]) > drop_rates).tolist()
    for idx, block in enumerate(visual.transformer.resblocks):
        if drop_flags[idx]:
            x = block(x)
        if idx + out_hidden_idx == num_blocks:
            return x.permute(1, 0, 2)[:, 0]

    if not visual.transformer.batch_first:
        x = x.transpose(0, 1)  # NLD -> LND

    pooled, _ = visual._pool(x)

    if visual.proj is not None:
        pooled = pooled @ visual.proj

    return pooled

