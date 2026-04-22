import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import math
import random
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTokenizer

import swanlab


test_prompt = "van gogh style, oil painting, Spring Festival temple fair in an old Chinese village, crowded people, red lanterns, lion dance, traditional stalls, festive atmosphere, swirling sky, thick impasto brushstrokes, rich details, post-impressionism"
generator = torch.Generator(device="cuda").manual_seed(42)

# =========================
# Dataset
# =========================
class SDDataset(Dataset):
    def __init__(self, json_path, tokenizer, size=512):
        self.data = [json.loads(line) for line in open(json_path, 'r', encoding='utf-8')]
        self.data = self.data * 10
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=512, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.05
            # ), 
            #I think delete it can learn Vangogh's color style better,maybe...
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)

        prompt = item["prompt"]

        # caption + token dropout
        if random.random() < 0.1:
            prompt = ""
        else:
            words = prompt.split(", ")
            words = [w for w in words if random.random() > 0.1]
            prompt = ", ".join(words)

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids[0]
        }

# =========================
# 2. config
# =========================
config = {
    "model_name": "runwayml/stable-diffusion-v1-5",
    "dataset": "/root/autodl-tmp/llama-diffusion/dataset/image_prompts.jsonl",
    "image_size": 512,
    "lora_rank": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["to_q", "to_v", "to_k", "to_out.0"],
    "lr": 1e-5,
    "batch_size": 16,
    "epochs": 20,
    "precision": "fp16",
    "lr_scheduler": "constant",
    "warmup_steps": 2000,
    "min_lr_ratio": 0.1,
}

swanlab.init(
    project="sd-lora-training",
    config=config,
    experiment_name="lora_with_scheduler"
)

# =========================
#  model
# =========================
model_id = config["model_name"]

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
vae.eval()
text_encoder.eval()
unet.train()

noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# =========================
#  LoRA
# =========================
lora_config = LoraConfig(
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    target_modules=config["target_modules"],
    lora_dropout=config["lora_dropout"],
    bias="none",
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# =========================
# data
# =========================
dataset = SDDataset(config["dataset"], tokenizer, size=config["image_size"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# =========================
# optimizer
# =========================
optimizer = torch.optim.AdamW(unet.parameters(), lr=config["lr"])
scaler = torch.cuda.amp.GradScaler()
os.makedirs("lora_output", exist_ok=True)

# =========================
# scheduler
# =========================
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

# =========================
#  training
# =========================
global_step = 0

for epoch in range(config["epochs"]):
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
        input_ids = batch["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

            noise = torch.randn_like(latents)
            noise = noise + 0.05 * torch.randn_like(latents)  #  noise offset

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0]

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            swanlab.log({
                "train/loss": loss.item(),
                "epoch": epoch
            }, step=global_step)

        global_step += 1

    # =========================
    # save checkpoint
    # =========================

    save_path = f"lora_model/64-1/epoch_{epoch}"
    unet.save_pretrained(save_path)
    print(f"Saved to {save_path}")

    # =========================
    # generate eval image
    # =========================
    pipe.unet = unet.to("cuda")

    with torch.no_grad():
        image = pipe(
            test_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

    os.makedirs(f"eval_output/64-1", exist_ok=True)
    image.save(f"eval_output/64-1/epoch_{epoch}.png")

    print(f"Saved eval image for epoch {epoch}")

print("Training finished!")
swanlab.finish()