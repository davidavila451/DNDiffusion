# train.py

import os
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator

# Load config
config_path = os.path.join(os.path.dirname(__file__), '../configs/training_config.yaml')

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

accelerator = Accelerator(mixed_precision=config["compute"]["mixed_precision"])
device = accelerator.device

# Load components
vae_path = config["model"]["vae_model"]
vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True).to(device)
tokenizer = CLIPTokenizer.from_pretrained(config["model"]["text_encoder"], local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(config["model"]["text_encoder"], local_files_only=True).to(device)

unet = UNet2DConditionModel(
    sample_size=config["model"]["unet_config"]["sample_size"],
    in_channels=config["model"]["unet_config"]["in_channels"],
    out_channels=config["model"]["unet_config"]["out_channels"],
    layers_per_block=config["model"]["unet_config"]["layers_per_block"],
    block_out_channels=tuple(config["model"]["unet_config"]["block_out_channels"]),
    down_block_types=tuple(config["model"]["unet_config"]["down_block_types"]),
    up_block_types=tuple(config["model"]["unet_config"]["up_block_types"]),
    cross_attention_dim=config["model"]["unet_config"]["cross_attention_dim"]
).to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config["optim"]["learning_rate"],
    weight_decay=config["optim"]["weight_decay"]
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Dataset Loader
def load_dnd_dataset():
    if config["dataset"]["name"].endswith(".json"):
        import json
        with open(config["dataset"]["name"], "r") as f:
            data = json.load(f)

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data, transform):
                self.data = data
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                image = self.transform(Image.open(item["image"]).convert("RGB"))
                return {
                    "image": image,
                    "text": item["text"]
                }

        transform = transforms.Compose([
            transforms.Resize(config["dataset"]["resolution"]),
            transforms.CenterCrop(config["dataset"]["resolution"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = CustomDataset(data, transform)
    else:
        dataset = load_dataset(config["dataset"]["name"], split="train")
        transform = transforms.Compose([
            transforms.Resize(config["dataset"]["resolution"]),
            transforms.CenterCrop(config["dataset"]["resolution"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        def preprocess(examples):
            examples["image"] = [transform(image.convert("RGB")) for image in examples["image"]]
            return examples

        dataset = dataset.with_transform(preprocess)

    return dataset

dataset = load_dnd_dataset()
dataloader = DataLoader(dataset, batch_size=config["training"]["train_batch_size"], shuffle=config["dataset"]["shuffle"])

# Enable memory optimizations
if config["compute"]["gradient_checkpointing"]:
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

vae.eval()  # VAE is frozen
unet.train()
text_encoder.train()

unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

global_step = 0

for epoch in range(1000):  # or until max steps
    for batch in tqdm(dataloader):
        with accelerator.accumulate(unet):
            pixel_values = batch["image"].to(device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Tokenize
            tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            encoder_hidden_states = text_encoder(**tokenized).last_hidden_state

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config["compute"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % config["training"]["checkpointing_steps"] == 0:
            accelerator.save_state(os.path.join(config["training"]["output_dir"], f"checkpoint-{global_step}"))

        if global_step >= config["training"]["max_train_steps"]:
            accelerator.print("Training complete.")
            accelerator.save_state(config["training"]["output_dir"])
            exit()
