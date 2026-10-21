import json
import os
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEMO_DIR = Path(__file__).parent.parent
CFPB_PROMPT = 'Classify this consumer financial complaint. Reply with JSON: {"product": ..., "issue": ...}'

# min_accuracy sits well below each model's full test score (93.4% intent, 77.8% product),
# so it only catches a broken merge, not ordinary sampling noise.
TASKS = {
    "banking77": {
        "adapter": "qwen2.5-1.5b-banking77",
        "prompt": "Classify the banking customer message into its intent label.",
        "test": "/data/test.jsonl",
        "stride": 150,
        "sample": 20,
        "card": "MODEL_CARD.md",
        "max_new_tokens": 20,
        "min_accuracy": 0.8,
    },
    "cfpb": {
        "adapter": "qwen2.5-1.5b-cfpb-n20000",
        "prompt": CFPB_PROMPT,
        "test": "/cfpb-data/tasks/cfpb/test.jsonl",
        "stride": 66,
        "sample": 50,
        "card": "MODEL_CARD_CFPB.md",
        "max_new_tokens": 80,
        "min_accuracy": 0.6,
    },
}

app = modal.App("banking77-publish")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "peft", "accelerate", "huggingface_hub")
    .add_local_file(DEMO_DIR / "data" / "test.jsonl", "/data/test.jsonl")
    .add_local_file(DEMO_DIR / "cfpb" / "task.py", "/root/cfpb_task.py")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("banking77-adapters")
cfpb_data = modal.Volume.from_name("tinytriage-data")
hf_token = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


def expected_and_answer(task: str, row: dict, raw: str) -> tuple[str, str]:
    if task == "banking77":
        return row["label"], raw
    import cfpb_task

    pred = cfpb_task.parse(raw)
    return row["product"], pred["product"] if pred else raw


def sanity_check(task: str, model, tokenizer) -> float:
    spec = TASKS[task]
    rows = [json.loads(line) for line in open(spec["test"])][:: spec["stride"]][: spec["sample"]]
    correct = 0
    for row in rows:
        messages = [{"role": "system", "content": spec["prompt"]}, {"role": "user", "content": row["text"]}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        output = model.generate(**inputs, max_new_tokens=spec["max_new_tokens"], do_sample=False)
        raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        expected, answer = expected_and_answer(task, row, raw)
        correct += answer == expected
        print(f"{'ok ' if answer == expected else 'BAD'}  {expected:55.55s} <- {answer}")
    return correct / len(rows)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters, "/cfpb-data": cfpb_data},
    secrets=[hf_token],
    timeout=60 * 60,
)
def merge_and_publish(task: str, repo_id: str, model_card: str, upload: bool) -> float:
    import shutil

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = f"/adapters/{TASKS[task]['adapter']}"
    out_dir = Path("/tmp/merged")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    accuracy = sanity_check(task, merged, tokenizer)
    print(f"sanity check accuracy: {accuracy:.0%}")
    if accuracy < TASKS[task]["min_accuracy"]:
        raise RuntimeError("merged model failed the sanity check; not publishing")

    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "lora").mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(f"{adapter_dir}/{name}", out_dir / "lora" / name)
    (out_dir / "README.md").write_text(model_card)

    print("files:", sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()))
    if upload:
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(out_dir), commit_message="Upload merged model and LoRA adapter")
        print(f"published https://huggingface.co/{repo_id}")
    return accuracy


@app.local_entrypoint()
def main(repo_id: str, task: str = "banking77", upload: bool = False):
    model_card = (Path(__file__).parent / TASKS[task]["card"]).read_text()
    merge_and_publish.remote(task, repo_id, model_card, upload)
