import os
from datetime import datetime
import torch
import soundfile as sf
from diffusers import StableAudioPipeline

# ---------- CONFIGURATION ----------
MODEL = "stabilityai/stable-audio-open-1.0"
OUTPUT_DIR = "./output/videos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableAudioPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
pipe = pipe.to(DEVICE)

# define the prompts
prompt = "The sound of a hammer hitting metal and a man saying 'Damn this is hard'."
negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator(DEVICE).manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"dnd_audio{timestamp}.wav"
filepath = os.path.join(OUTPUT_DIR, filename)
sf.write(filepath, output, pipe.vae.sampling_rate)