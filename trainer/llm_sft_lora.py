import os
import logging
import torch
import swanlab   
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from modelscope import snapshot_download
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
# ==================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "LLM-Research/Llama-3.2-3B-Instruct"
DATA_PATH = "/root/autodl-tmp/llama-diffusion/dataset/llm_prompt_sft.json"
OUTPUT_DIR = "/root/autodl-tmp/llama-diffusion/outs/llama32-sft-final"
class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            import swanlab
            swanlab.log(logs)
def main():
    set_seed(42)

    swanlab.init(
        project="llama_diffusion_project",
        experiment_name="llama3.2-3b-sft-lora",
        config={
            "model": MODEL_NAME,
            "lr": 2e-5,
            "batch_size": 2,
            "grad_accum": 8,
            "epochs": 3,
            "lora_r": 8,
            "lora_alpha": 32,
        }
    )

    logger.info("loading model...")
    model_dir = snapshot_download(MODEL_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    logger.info("loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    dataset = dataset.train_test_split(test_size=0.1)

    def format_func(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }
    
    dataset = dataset.map(format_func)
    def tokenize_func(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        return tokenized

    dataset = dataset.map(tokenize_func, remove_columns=dataset["train"].column_names)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        report_to="none",   
        save_total_limit=2,
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        callbacks=[SwanLabCallback()],
    )

    logger.info("Starting training...")

    for step, output in enumerate(trainer.train()):
        if step % 10 == 0:
            swanlab.log({
                "step": step,
                "loss": output.training_loss if hasattr(output, "training_loss") else None,
            })


    final_path = f"{OUTPUT_DIR}/final_model"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)


    swanlab.finish()

    logger.info(f"All Done.Model saved to: {final_path}")

if __name__ == "__main__":
    main()