from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "a dwarven warrior in golden armor, standing in a misty battlefield"
image = pipe(prompt).images[0]
image.save("output/dwarf_warrior.png")