import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
import os
from peft import PeftModel
unet_lora = "/root/autodl-tmp/llama-diffusion/trainer/lora_output/epoch_599"
class SDPipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, unet_lora, torch_dtype=torch.float16).to("cuda")
        self.pipe.unet.set_adapter("default")
        for module in self.pipe.unet.modules():
            if hasattr(module, "scaling"):
                for k in module.scaling.keys():
                    module.scaling[k] = 0.6
        self.pipe.enable_attention_slicing()

    def generate(self, prompt, save_dir="/root/autodl-tmp/llama-diffusion/out"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{timestamp}.png")

        image = self.pipe(prompt).images[0]
        image.save(save_path)

        return save_path