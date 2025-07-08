import os
from datetime import datetime
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# ---------- CONFIGURATION ----------
MODEL = "stabilityai/stable-audio-open-1.0"
OUTPUT_DIR = "./output/sounds"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Models")

# Download model
model, model_config = get_pretrained_model(MODEL)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(DEVICE)
print("Models loaded")
# define the prompts
prompt = "The sound of a hammer hitting metal and a man saying 'Damn this is hard'."

print("Setting up conditioning")
# Set up text and timing conditioning
conditioning = [{
    "prompt": prompt,
    "seconds_start": 0, 
    "seconds_total": 30
}]

print("Conditions set")
print("Running generation")
# run the generation
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=DEVICE
)

print("Generation complete")
print("Rearragning audio")
# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

print("Rearrange complete")
print("Saving audio")
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"dnd_audio{timestamp}.wav"
filepath = os.path.join(OUTPUT_DIR, filename)
torchaudio.save(filepath, output, sample_rate)