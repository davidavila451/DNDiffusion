import os
from datetime import datetime
from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image

# ---------- CONFIGURATION ----------
MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
OUTPUT_DIR = "./output/videos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "./output/images/dnd_image_20250625_115642.png"

# ---------- SETUP ----------
print("Loading models")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    MODEL,
    variant="fp16"
).to(DEVICE)
print("Models loaded")

# ---------- GENERATION ----------
print("Starting generation")
image = Image.open(IMAGE_PATH).convert("RGB").resize((576, 1024))
video_frames = pipe(image, decode_chunk_size=8, num_frames=25).frames

print("Generation Complete")

print("Saving gif")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"dnd_video_{timestamp}.gif"
filepath = os.path.join(OUTPUT_DIR, filename)

video_frames[0].save(filepath, save_all=True, append_images=video_frames[1:], duration=80, loop=0)
print("Gif saved")