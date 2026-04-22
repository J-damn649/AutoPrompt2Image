import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMAgent:
    def __init__(self, model_path="/root/autodl-tmp/llama-diffusion/models/merged_model"):
        self.model_dir = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages, max_new_tokens=256):

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if "<|assistant|>" in result:
            return result.split("<|assistant|>")[-1].strip()
        else:
            return result.strip()


if __name__ == "__main__":

    agent = LLMAgent()

    messages = [
        {
            "role": "system",
            "content": """You are a professional prompt engineer for Stable Diffusion.
Convert user input into a detailed prompt including:
- style
- lighting
- quality
Only output the final prompt."""
        },
        {
            "role": "user",
            "content": "draw a cat in the style of cyberpunk"
        }
    ]

    output = agent.generate(messages)

    print("\n=== Enhanced Prompt ===\n")
    print(output)