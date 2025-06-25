import os
from datetime import datetime
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline, StableCascadeUNet
from transformers import AutoTokenizer
import torch

# ---------- CONFIGURATION ----------
PRIOR_MODEL = "stabilityai/stable-cascade-prior"
DECODER_MODEL = "stabilityai/stable-cascade"
OUTPUT_DIR = "./output/images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- SETUP ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading models...")
prior_unet = StableCascadeUNet.from_pretrained(
    PRIOR_MODEL,
    subfolder="prior_lite"
)

decoder_unet = StableCascadeUNet.from_pretrained(
    DECODER_MODEL,
    subfolder="decoder_lite"
)

prior = StableCascadePriorPipeline.from_pretrained(
    PRIOR_MODEL, 
    prior=prior_unet
)


decoder = StableCascadeDecoderPipeline.from_pretrained(
    DECODER_MODEL, 
    decoder=decoder_unet
)


print("Models loaded and ready.")

# ---------- USER LOOP ----------
def main():
    while True:
        prompt = input("\nEnter your D&D fantasy prompt (or 'exit' to quit): ").strip()
        negative_prompt = ""
        if prompt.lower() == "exit":
            print("Goodbye!")
            return

        print("Generating image from prompt...")

        print("Starting prior")
        prior.to(DEVICE)
        prior_output = prior(
            prompt=prompt,
            height=1024,
            width=1024,
            negative_prompt=negative_prompt,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            num_inference_steps=20
        )
        print("Starting decoder")
        decoder.to(DEVICE)
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=10
        ).images[0]

        print("Saving image")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dnd_image_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        decoder_output.save(filepath)

        print(f"✅ Image saved to: {filepath}")

if __name__ == "__main__":
    main()