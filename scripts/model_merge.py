from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "/root/autodl-tmp/modelscope/hub/models/LLM-Research/Llama-3.2-3B-Instruct"
lora_path = "/root/autodl-tmp/llama-diffusion/models/llama32-sft-final/final_model"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# get lora model
model = PeftModel.from_pretrained(model, lora_path)

# merge lora weights into base model
model = model.merge_and_unload()

save_path = "/root/autodl-tmp/llama-diffusion/models/merged_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model merged and saved to:", save_path)