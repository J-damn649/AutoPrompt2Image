from datasets import load_dataset
import os
ds = load_dataset("uumlaut/VanGoghPaintings", split="train")

save_dir = "/root/autodl-tmp/llama-diffusion/dataset/vangogh_data"
os.makedirs(save_dir, exist_ok=True)

for i, item in enumerate(ds):
    img = item["image"]
    img.save(f"{save_dir}/{i:04d}.jpg")