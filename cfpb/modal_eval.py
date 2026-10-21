"""Evaluate CFPB models on Modal. Fine-tuned models: vLLM runs inside the job, so nothing stays up afterwards."""

import json
import os
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
L4_HOURLY_USD = 0.80
FRONTIER_MODELS = {"claude-haiku-5.5": "anthropic/claude-haiku-5-5", "claude-sonnet-5.5": "anthropic/claude-sonnet-5-5"}
FRONTIER_SAMPLE = 1000
HERE = Path(__file__).resolve().parent
LOCAL_RESULTS = HERE.parent / "results" / "cfpb"

app = modal.App("tinytriage-cfpb-eval")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("banking77-adapters")
data = modal.Volume.from_name("tinytriage-data")
perplexity = modal.Secret.from_dict({"PERPLEXITY_API_KEY": os.environ.get("PERPLEXITY_API_KEY", "")})

source = {"ignore": ["__pycache__", "modal_app.py", "modal_eval.py"]}
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "openai")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTHONPATH": "/root/cfpb"})
    .add_local_dir(HERE, "/root/cfpb", **source)
)
client_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("openai")
    .env({"PYTHONPATH": "/root/cfpb"})
    .add_local_dir(HERE, "/root/cfpb", **source)
)

TASK_DIR = Path("/data/tasks/cfpb")


def save(result: dict, limit: int) -> dict:
    suffix = f"_smoke{limit}" if limit else ""
    out = TASK_DIR / "results" / f"eval_{result['summary']['name']}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    return result["summary"]


def wait_for_server(url: str, timeout_s: int = 900) -> None:
    import time
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError("vLLM did not start in time")


@app.function(
    image=vllm_image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters, "/data": data},
    timeout=3 * 60 * 60,
)
def eval_finetuned(limit: int = 0, only: str = "", include_base: bool = True, split: str = "test") -> list[dict]:
    import asyncio
    import subprocess

    import evaluate
    import task as cfpb_task
    from openai import AsyncOpenAI

    names = sorted(p.name for p in Path("/adapters").glob("qwen2.5-1.5b-cfpb-n*"))
    if only:
        names = [n for n in names if n in only.split(",")]
    lora_args = [f"{n}=/adapters/{n}" for n in names]
    server = subprocess.Popen([
        "vllm", "serve", BASE_MODEL, "--served-model-name", "base",
        "--enable-lora", "--lora-modules", *lora_args, "--max-loras", str(len(names)), "--max-lora-rank", "16",
        "--max-model-len", "4096", "--port", "8000",
    ])
    try:
        wait_for_server("http://localhost:8000/v1/models")
        labels = json.loads((TASK_DIR / "labels.json").read_text())
        test = evaluate.load_jsonl(TASK_DIR / f"{split}.jsonl")
        test = test[:limit] if limit else test
        suffix = "" if split == "test" else f"_{split}"
        runs = [(n, n, cfpb_task.SHORT_PROMPT) for n in names]
        if include_base:
            runs.insert(0, ("qwen1.5b-base", "base", cfpb_task.full_prompt(labels)))

        async def run_all() -> list[dict]:
            client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="local")
            summaries = []
            for name, model, system in runs:
                outputs, wall = await evaluate.run_model(client, model, test, system, "chat", concurrency=32)
                summaries.append(save(evaluate.score(name + suffix, test, outputs, labels, wall, L4_HOURLY_USD), limit))
                print(name, f"pair {summaries[-1]['pair_accuracy']:.1%}")
            return summaries

        summaries = asyncio.run(run_all())
        data.commit()
        return summaries
    finally:
        server.terminate()


@app.function(image=client_image, volumes={"/data": data}, secrets=[perplexity], cpu=2, timeout=2 * 60 * 60)
def eval_frontier(limit: int = 0, only: str = "") -> list[dict]:
    import asyncio

    import evaluate
    import task as cfpb_task
    from openai import AsyncOpenAI

    labels = json.loads((TASK_DIR / "labels.json").read_text())
    sample = evaluate.load_jsonl(TASK_DIR / "test.jsonl")[: limit or FRONTIER_SAMPLE]
    async def run_all() -> list[dict]:
        client = AsyncOpenAI(base_url="https://api.perplexity.ai/v1", api_key=os.environ["PERPLEXITY_API_KEY"], max_retries=5)
        summaries = []
        for name, model in FRONTIER_MODELS.items():
            if only and name != only:
                continue
            outputs, wall = await evaluate.run_model(
                client, model, sample, cfpb_task.full_prompt(labels), "responses", concurrency=16
            )
            summaries.append(save(evaluate.score(name, sample, outputs, labels, wall, None), limit))
            print(name, f"pair {summaries[-1]['pair_accuracy']:.1%}")
        return summaries

    summaries = asyncio.run(run_all())
    data.commit()
    return summaries


@app.function(image=client_image, volumes={"/data": data}, cpu=2, memory=4096, timeout=10 * 60)
def report() -> str:
    import report as step

    return step.build(TASK_DIR / "results", FRONTIER_SAMPLE)


def print_table(summaries: list[dict]) -> None:
    for s in summaries:
        cost = s["cost_per_1k_requests_usd"]
        print(f"{s['name']:28s} n={s['n']:5d}  product {s['product_accuracy']:6.1%}  pair {s['pair_accuracy']:6.1%}  "
              f"invalid {s['invalid_label_rate']:5.1%}  reasoning {s['avg_reasoning_tokens']:5.0f}  p50 {s['latency_p50_s']:.2f}s  "
              f"cost/1k {'n/a' if cost is None else f'${cost:.4f}'}")


@app.local_entrypoint()
def main(step: str = "finetuned", limit: int = 0, only: str = "", include_base: bool = True, split: str = "test"):
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    if step == "report":
        text = report.remote()
        (LOCAL_RESULTS / "report.md").write_text(text + "\n")
        print(text)
        return
    fn = {"finetuned": eval_finetuned, "frontier": eval_frontier}[step]
    summaries = fn.remote(limit, only) if step == "frontier" else fn.remote(limit, only, include_base, split)
    if limit:
        print_table(summaries)
        return
    tag = "".join(f"_{part}" for part in (only, "" if split == "test" else split) if part)
    (LOCAL_RESULTS / f"eval_summary_{step}{tag}.json").write_text(json.dumps(summaries, indent=2))
    print_table(summaries)
