import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

DEMO_DIR = Path(__file__).parent.parent
LABELS: list[str] = json.loads((DEMO_DIR / "data" / "labels.json").read_text())
LABEL_SET = set(LABELS)
LABEL_BY_LOWER = {label.lower(): label for label in LABELS}
MAX_OUTPUT_TOKENS = 40

# "short" must match FINETUNE_SYSTEM_PROMPT in data/prepare.py: the fine-tuned model
# only knows the labels from training, while prompted models need the full list.
PROMPTS = {
    "short": "Classify the banking customer message into its intent label.",
    "full": (
        "You classify banking customer support messages by intent. "
        "Reply with exactly one label from this list and nothing else:\n" + "\n".join(LABELS)
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a model on the Banking77 test set.")
    p.add_argument("--name", required=True, help="results/<name>.json")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", required=True, help="env var holding the API key")
    p.add_argument("--api", choices=["chat", "responses"], default="chat", help="responses for Perplexity Agent API")
    p.add_argument(
        "--reasoning-effort",
        help="responses API only; reasoning models can spend the whole token budget thinking and return no label",
    )
    p.add_argument("--prompt", choices=PROMPTS, required=True)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--logprobs", action="store_true", help="record answer confidence (chat API only)")
    p.add_argument("--limit", type=int, help="score only the first N test examples")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens (API models)")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens (API models)")
    p.add_argument("--gpu-hourly", type=float, help="USD per GPU-hour (self-hosted models)")
    return p.parse_args()


def normalize(answer: str) -> str:
    # Models sometimes append commentary after the label, so only the first line counts.
    first_line = answer.strip().split("\n")[0]
    cleaned = first_line.strip().strip("`'\".").strip().replace(" ", "_")
    return LABEL_BY_LOWER.get(cleaned.lower(), cleaned)


async def call_chat(client: AsyncOpenAI, model: str, messages: list[dict], logprobs: bool) -> dict:
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=MAX_OUTPUT_TOKENS, logprobs=logprobs or None
    )
    choice = resp.choices[0]
    result = {
        "raw": choice.message.content or "",
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    if logprobs and choice.logprobs:
        result["confidence"] = math.exp(sum(t.logprob for t in choice.logprobs.content))
    return result


async def call_responses(client: AsyncOpenAI, model: str, messages: list[dict], reasoning_effort: str | None) -> dict:
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None
    resp = await client.responses.create(
        model=model, input=messages, max_output_tokens=MAX_OUTPUT_TOKENS, reasoning=reasoning
    )
    extra = resp.usage.model_extra or {}
    cost = extra.get("cost") or {}
    return {
        "raw": resp.output_text or "",
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "reported_cost_usd": cost.get("total_cost"),
        "tool_calls": extra.get("tool_calls_details"),
    }


async def classify(client: AsyncOpenAI, args: argparse.Namespace, system: str, text: str, sem: asyncio.Semaphore) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    async with sem:
        start = time.perf_counter()
        if args.api == "responses":
            result = await call_responses(client, args.model, messages, args.reasoning_effort)
        else:
            result = await call_chat(client, args.model, messages, args.logprobs)
        result["latency_s"] = time.perf_counter() - start
    return result


def summarize(args: argparse.Namespace, rows: list[dict], wall_s: float) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    invalid = sum(r["pred"] not in LABEL_SET for r in rows)
    latencies = sorted(r["latency_s"] for r in rows)
    input_tokens = sum(r["input_tokens"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)

    reported = [r.get("reported_cost_usd") for r in rows]
    cost_total = None
    if all(c is not None for c in reported):
        cost_total = sum(reported)
    elif args.price_in is not None and args.price_out is not None:
        cost_total = (input_tokens * args.price_in + output_tokens * args.price_out) / 1e6
    elif args.gpu_hourly is not None:
        cost_total = args.gpu_hourly * wall_s / 3600

    by_label = defaultdict(lambda: [0, 0])
    for r in rows:
        by_label[r["label"]][0] += r["correct"]
        by_label[r["label"]][1] += 1
    per_label = {label: c / t for label, (c, t) in by_label.items()}

    return {
        "name": args.name,
        "model": args.model,
        "prompt": args.prompt,
        "split": args.split,
        "n": n,
        "accuracy": correct / n,
        "invalid_label_rate": invalid / n,
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": latencies[int(0.95 * (n - 1))],
        "avg_input_tokens": input_tokens / n,
        "avg_output_tokens": output_tokens / n,
        "wall_time_s": wall_s,
        "concurrency": args.concurrency,
        "cost_per_1k_requests_usd": None if cost_total is None else cost_total / n * 1000,
        "requests_with_tool_calls": sum(bool(r.get("tool_calls")) for r in rows),
        "worst_labels": sorted(per_label.items(), key=lambda kv: kv[1])[:10],
    }


def load_split(split: str) -> list[dict]:
    rows = [json.loads(line) for line in (DEMO_DIR / "data" / f"{split}.jsonl").open()]
    if split == "test":
        return rows
    return [{"text": r["prompt"][1]["content"], "label": r["completion"][0]["content"]} for r in rows]


async def main() -> None:
    args = parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env], max_retries=5)
    test = load_split(args.split)
    if args.limit:
        test = test[: args.limit]

    system = PROMPTS[args.prompt]
    sem = asyncio.Semaphore(args.concurrency)

    await classify(client, args, system, test[0]["text"], sem)

    start = time.perf_counter()
    results = await asyncio.gather(*(classify(client, args, system, ex["text"], sem) for ex in test))
    wall_s = time.perf_counter() - start

    rows = []
    for ex, res in zip(test, results):
        pred = normalize(res["raw"])
        rows.append({**ex, **res, "pred": pred, "correct": pred == ex["label"]})

    summary = summarize(args, rows, wall_s)
    out = DEMO_DIR / "results" / f"{args.name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "worst_labels"}, indent=2))
    print("worst labels:")
    for label, acc in summary["worst_labels"][:5]:
        print(f"  {acc:6.1%}  {label}")
    print(f"saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
