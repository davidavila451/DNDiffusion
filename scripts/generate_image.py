from diffusers import StableCascadeCombinedPipeline
import torch

pipe = StableCascadeCombinedPipeline.from_pretrained(
    "davidavila451/dnd-model-SD",
    torch_dtype=torch.float16
).to("cuda")