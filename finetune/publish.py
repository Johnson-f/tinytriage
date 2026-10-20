import json
import os
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "qwen2.5-1.5b-banking77"
SYSTEM_PROMPT = "Classify the banking customer message into its intent label."
DEMO_DIR = Path(__file__).parent.parent

app = modal.App("banking77-publish")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "peft", "accelerate", "huggingface_hub")
    .add_local_file(DEMO_DIR / "data" / "test.jsonl", "/data/test.jsonl")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("banking77-adapters")
hf_token = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


def sanity_check(model, tokenizer, n: int = 20) -> float:
    rows = [json.loads(line) for line in open("/data/test.jsonl")][::150][:n]
    correct = 0
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["text"]}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        correct += answer == row["label"]
        print(f"{'ok ' if answer == row['label'] else 'BAD'}  {row['label']:45s} <- {answer}")
    return correct / len(rows)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters},
    secrets=[hf_token],
    timeout=60 * 60,
)
def merge_and_publish(repo_id: str, model_card: str, upload: bool) -> float:
    import shutil

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = f"/adapters/{ADAPTER_NAME}"
    out_dir = Path("/tmp/merged")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    accuracy = sanity_check(merged, tokenizer)
    print(f"sanity check accuracy: {accuracy:.0%}")
    # The full test-set score is 93.4%; a much lower score here means the merge went wrong.
    if accuracy < 0.8:
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
def main(repo_id: str, upload: bool = False):
    model_card = (Path(__file__).parent / "MODEL_CARD.md").read_text()
    merge_and_publish.remote(repo_id, model_card, upload)
