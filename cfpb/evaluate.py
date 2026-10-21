"""Score models on CFPB complaint routing: product and issue as JSON."""

import asyncio
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from openai import APIStatusError, AsyncOpenAI

import task as cfpb_task

MAX_OUTPUT_TOKENS = 80
# Claude reasons on long, many-label prompts even with reasoning "none" and returns nothing,
# so frontier models run with "high" reasoning and enough output budget left for the JSON.
RESPONSES_MAX_OUTPUT_TOKENS = 4096
RESPONSES_REASONING_EFFORT = "high"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


FATAL_STATUS = {401, 402, 403}


async def ask(client: AsyncOpenAI, model: str, system: str, text: str, api: str, sem: asyncio.Semaphore) -> dict:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    async with sem:
        start = time.perf_counter()
        try:
            result = await (ask_responses if api == "responses" else ask_chat)(client, model, messages)
        except APIStatusError as e:
            # Out of credits or bad keys should stop the run instead of quietly scoring empty answers.
            if e.status_code in FATAL_STATUS:
                raise
            result = {"raw": "", "input_tokens": 0, "output_tokens": 0, "error": f"HTTP {e.status_code}"}
        result["latency_s"] = time.perf_counter() - start
    return result


async def ask_responses(client: AsyncOpenAI, model: str, messages: list[dict]) -> dict:
    resp = await client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=RESPONSES_MAX_OUTPUT_TOKENS,
        reasoning={"effort": RESPONSES_REASONING_EFFORT},
    )
    usage = resp.usage
    if usage is None:
        return {"raw": resp.output_text or "", "input_tokens": 0, "output_tokens": 0, "error": "no usage reported"}
    extra = usage.model_extra or {}
    details = usage.output_tokens_details
    return {
        "raw": resp.output_text or "",
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": (extra.get("cost") or {}).get("total_cost"),
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
    }


async def ask_chat(client: AsyncOpenAI, model: str, messages: list[dict]) -> dict:
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=MAX_OUTPUT_TOKENS, logprobs=True
    )
    choice = resp.choices[0]
    tokens = choice.logprobs.content if choice.logprobs else []
    return {
        "raw": choice.message.content or "",
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "confidence": math.exp(sum(t.logprob for t in tokens)) if tokens else None,
    }


async def run_model(client: AsyncOpenAI, model: str, rows: list[dict], system: str, api: str, concurrency: int) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(concurrency)
    await ask(client, model, system, rows[0]["text"], api, sem)
    start = time.perf_counter()
    outputs = await asyncio.gather(*(ask(client, model, system, r["text"], api, sem) for r in rows))
    return outputs, time.perf_counter() - start


def score(name: str, rows: list[dict], outputs: list[dict], labels: dict[str, list[str]], wall_s: float, gpu_hourly: float | None) -> dict:
    scored = []
    for row, out in zip(rows, outputs):
        pred = cfpb_task.parse(out["raw"])
        valid_label = bool(pred) and pred["issue"] in labels.get(pred["product"], [])
        scored.append({
            **row,
            **out,
            "pred": pred,
            "valid_json": pred is not None,
            "valid_label": valid_label,
            "product_correct": bool(pred) and pred["product"] == row["product"],
            "pair_correct": bool(pred) and pred["product"] == row["product"] and pred["issue"] == row["issue"],
        })

    n = len(scored)
    latencies = sorted(s["latency_s"] for s in scored)
    by_product = defaultdict(list)
    for s in scored:
        by_product[s["product"]].append(s["pair_correct"])

    reported = [s["cost_usd"] for s in scored if s.get("cost_usd") is not None]
    if gpu_hourly is not None:
        cost_total, priced = gpu_hourly * wall_s / 3600, n
    elif reported:
        cost_total, priced = sum(reported), len(reported)
    else:
        cost_total, priced = None, 0

    summary = {
        "name": name,
        "n": n,
        "product_accuracy": sum(s["product_correct"] for s in scored) / n,
        "pair_accuracy": sum(s["pair_correct"] for s in scored) / n,
        "invalid_json_rate": sum(not s["valid_json"] for s in scored) / n,
        "invalid_label_rate": sum(not s["valid_label"] for s in scored) / n,
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": latencies[int(0.95 * (n - 1))],
        "avg_input_tokens": sum(s["input_tokens"] for s in scored) / n,
        "avg_output_tokens": sum(s["output_tokens"] for s in scored) / n,
        "avg_reasoning_tokens": sum(s.get("reasoning_tokens", 0) for s in scored) / n,
        "wall_time_s": wall_s,
        "cost_per_1k_requests_usd": None if cost_total is None else cost_total / priced * 1000,
        "requests_failed": sum(bool(s.get("error")) for s in scored),
        "requests_missing_cost": n - priced,
        "pair_accuracy_by_product": {p: sum(v) / len(v) for p, v in sorted(by_product.items())},
    }
    return {"summary": summary, "rows": scored}
