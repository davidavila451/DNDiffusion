import os
from datetime import datetime
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
from transformers import AutoTokenizer
import torch

# ---------- CONFIGURATION ----------
PRIOR_MODEL = "stabilityai/stable-cascade-prior"
DECODER_MODEL = "stabilityai/stable-cascade"
OUTPUT_DIR = "../output/images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- SETUP ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(PRIOR_MODEL)

prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior", 
    variant="bf16", 
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)


decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade", 
    variant="bf16", 
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)


print("Models loaded and ready.")

# ---------- USER LOOP ----------
def main():
    while True:
        prompt = input("\nEnter your D&D fantasy prompt (or 'exit' to quit): ").strip()
        negative_prompt = ""
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        print("Generating image from prompt...")

        with torch.no_grad():
            prior.enable_model_cpu_offload()
            prior_output = prior(
                prompt=prompt,
                height=1024,
                width=1024,
                negative_prompt=negative_prompt,
                guidance_scale=4.0,
                num_images_per_prompt=1,
                num_inference_steps=20
            )
            decoder.enable_model_cpu_offload()
            decoder_output = decoder(
                image_embeddings=prior_output.image_embeddings.to(torch.float16),
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=0.0,
                output_type="pil",
                num_inference_steps=10
            ).images[0]
            decoder_output.save("{prompt}.png")

