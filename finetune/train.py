import json
import math
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
    .add_local_file(DEMO_DIR / "cfpb" / "task.py", "/root/cfpb_task.py")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("banking77-adapters", create_if_missing=True)
cfpb_data = modal.Volume.from_name("tinytriage-data", create_if_missing=True)


def cfpb_epochs(train_size: int, batch_size: int) -> int:
    # Small training sets need more passes to take enough optimizer steps (~200);
    # capped so 100 examples don't get hundreds of epochs.
    return min(20, max(2, math.ceil(200 * batch_size / train_size)))


def load_cfpb(train_size: int) -> tuple[list[dict], list[dict]]:
    import cfpb_task

    def to_example(row: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": cfpb_task.SHORT_PROMPT},
                {"role": "user", "content": row["text"]},
            ],
            "completion": [{"role": "assistant", "content": cfpb_task.completion(row)}],
        }

    base = Path("/cfpb-data/tasks/cfpb")
    pool = [json.loads(line) for line in (base / "train_pool.jsonl").open()]
    val = [json.loads(line) for line in (base / "val.jsonl").open()]
    return [to_example(r) for r in pool[:train_size]], [to_example(r) for r in val]


VOLUMES = {"/root/.cache/huggingface": hf_cache, "/adapters": adapters, "/cfpb-data": cfpb_data}


@app.function(image=image, gpu="L40S", volumes=VOLUMES, timeout=3 * 60 * 60)
def train(task: str, epochs: int, train_size: int) -> dict:
    return run_training(task, epochs, train_size)


def run_training(task: str, epochs: int, train_size: int) -> dict:
    import shutil

    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class CommitCheckpoints(TrainerCallback):
        # Checkpoints live on the volume so an interrupted run can resume instead of restarting.
        def on_save(self, args, state, control, **kwargs):
            adapters.commit()

    if task == "banking77":
        train_ds = load_dataset("json", data_files="/data/train.jsonl", split="train")
        val_ds = load_dataset("json", data_files="/data/val.jsonl", split="train")
        adapter_name, batch_size, max_length, eval_steps = ADAPTER_NAME, 32, 256, 100
        keep_best = False
    else:
        train_rows, val_rows = load_cfpb(train_size)
        train_ds, val_ds = Dataset.from_list(train_rows), Dataset.from_list(val_rows)
        adapter_name, batch_size, max_length = f"qwen2.5-1.5b-cfpb-n{train_size}", 16, 1024
        epochs = epochs or cfpb_epochs(train_size, batch_size)
        eval_steps = max(10, math.ceil(len(train_rows) * epochs / batch_size) // 10)
        # Small CFPB sets overfit within a few epochs, so keep the checkpoint with the lowest validation loss.
        keep_best = True

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    checkpoint_dir = Path(f"/adapters/checkpoints/{adapter_name}") if keep_best else Path("/tmp/checkpoints")
    config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_length=max_length,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps" if keep_best else "no",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=keep_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[CommitCheckpoints()] if keep_best else None,
    )
    resume = keep_best and any(checkpoint_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)

    trainer.save_model(f"/adapters/{adapter_name}")
    tokenizer.save_pretrained(f"/adapters/{adapter_name}")
    result = {
        "adapter": adapter_name,
        "epochs": epochs,
        "train_size": len(train_ds),
        "resumed": resume,
        "gpu": torch.cuda.get_device_name(0),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "history": trainer.state.log_history,
    }
    log_dir = Path("/adapters/logs")
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{adapter_name}.json").write_text(json.dumps(result, indent=2))
    if keep_best:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    adapters.commit()
    return result


def print_history(history: list[dict]) -> None:
    for entry in history:
        if "loss" in entry:
            print(f"  step {entry['step']:5d}  train_loss {entry['loss']:.4f}")
        if "eval_loss" in entry:
            print(f"  step {entry['step']:5d}  val_loss   {entry['eval_loss']:.4f}")


@app.local_entrypoint()
def main(task: str = "banking77", epochs: int = 0, sizes: str = "20000"):
    if task == "banking77":
        result = train.remote("banking77", epochs or 2, 0)
        out = DEMO_DIR / "results" / "train_log.json"
        out.write_text(json.dumps(result["history"], indent=2))
        print_history(result["history"])
        print(f"saved {out}")
        return

    out_dir = DEMO_DIR / "results" / "cfpb"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [(task, epochs, int(size)) for size in sizes.split(",")]
    for result in train.starmap(runs):
        out = out_dir / f"train_log_n{result['train_size']}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"{result['adapter']}: {result['train_size']} examples, {result['epochs']} epochs, "
              f"best val_loss {result['best_eval_loss']} at {result['best_checkpoint']}")
        print_history([h for h in result["history"] if "eval_loss" in h])
