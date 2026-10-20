import json
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "qwen2.5-1.5b-banking77"
DEMO_DIR = Path(__file__).parent.parent

app = modal.App("banking77-finetune")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "trl", "peft", "datasets", "accelerate")
    .add_local_dir(DEMO_DIR / "data", "/data")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("banking77-adapters", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters},
    timeout=60 * 60,
)
def train(epochs: int) -> list[dict]:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_ds = load_dataset("json", data_files="/data/train.jsonl", split="train")
    val_ds = load_dataset("json", data_files="/data/val.jsonl", split="train")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    config = SFTConfig(
        output_dir="/tmp/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_length=256,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()

    trainer.save_model(f"/adapters/{ADAPTER_NAME}")
    tokenizer.save_pretrained(f"/adapters/{ADAPTER_NAME}")
    adapters.commit()
    return trainer.state.log_history


@app.local_entrypoint()
def main(epochs: int = 2):
    history = train.remote(epochs)
    out = DEMO_DIR / "results" / "train_log.json"
    out.write_text(json.dumps(history, indent=2))

    for entry in history:
        if "loss" in entry:
            print(f"step {entry['step']:4d}  train_loss {entry['loss']:.4f}")
        if "eval_loss" in entry:
            print(f"step {entry['step']:4d}  val_loss   {entry['eval_loss']:.4f}")
    print(f"saved {out}")
