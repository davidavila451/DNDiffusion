from diffusers import StableCascadeCombinedPipeline
import torch

pipe = StableCascadeCombinedPipeline.from_pretrained(
    "davidavila451/dnd-model-SD",
    torch_dtype=torch.float16
).to("cpu")

#prompt = "a dwarven warrior in golden armor, standing in a misty battlefield"
#image = pipe(prompt).images[0]
#image.save("output/dwarf_warrior.png")