"""Build the CFPB results tables from saved evaluation runs: comparison, data-size curve, confidence routing."""

import json
from pathlib import Path

FRONTIER = ["claude-sonnet-5.5", "claude-haiku-5.5"]
SIZES = [100, 500, 1000, 5000, 20000]
ROUTED = [5000, 20000]
TARGETS = {"product_correct": [0.85, 0.90, 0.95], "pair_correct": [0.60, 0.70, 0.80, 0.90]}


def adapter(size: int) -> str:
    return f"qwen2.5-1.5b-cfpb-n{size}"


def load(results: Path, name: str) -> dict:
    return json.loads((results / f"eval_{name}.json").read_text())


def rate(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def comparison(results: Path, n: int) -> list[str]:
    lines = [
        "| Model | Product | Product + issue | Invalid | Avg input tokens | p50 latency | Cost / 1k |",
        "|---|---|---|---|---|---|---|",
    ]
    names = [adapter(20000), *FRONTIER, "qwen1.5b-base"]
    for name in names:
        run = load(results, name)
        rows, summary = run["rows"][:n], run["summary"]
        cost = summary["cost_per_1k_requests_usd"]
        lines.append(
            f"| {name} | {rate(rows, 'product_correct'):.1%} | {rate(rows, 'pair_correct'):.1%} "
            f"| {1 - rate(rows, 'valid_label'):.1%} | {summary['avg_input_tokens']:.0f} "
            f"| {summary['latency_p50_s']:.2f}s | ${cost:.4f} |"
        )
    return lines


def data_curve(results: Path, n: int) -> list[str]:
    baseline = {b["train_size"]: b for b in json.loads((results / "baseline_tfidf.json").read_text())}
    lines = [
        "| Training examples | Fine-tuned: product | Fine-tuned: product + issue | TF-IDF: product | TF-IDF: product + issue |",
        "|---|---|---|---|---|",
    ]
    base_rows = load(results, "qwen1.5b-base")["rows"][:n]
    lines.append(f"| 0 (base model) | {rate(base_rows, 'product_correct'):.1%} | {rate(base_rows, 'pair_correct'):.1%} | | |")
    for size in SIZES:
        rows = load(results, adapter(size))["rows"][:n]
        b = baseline[size]
        lines.append(
            f"| {size:,} | {rate(rows, 'product_correct'):.1%} | {rate(rows, 'pair_correct'):.1%} "
            f"| {b['sample1000_product_accuracy']:.1%} | {b['sample1000_pair_accuracy']:.1%} |"
        )
    return lines


def lowest_threshold_for(rows: list[dict], key: str, target: float) -> float | None:
    # Walk from most to least confident; the last point still meeting the target gives the most automation.
    best, correct = None, 0
    for i, r in enumerate(rows, start=1):
        correct += r[key]
        if correct / i >= target:
            best = r["confidence"]
    return best


def routing(results: Path, size: int, key: str) -> list[str]:
    by_confidence = lambda rows: sorted(rows, key=lambda r: r["confidence"], reverse=True)
    val = by_confidence(load(results, f"{adapter(size)}_val")["rows"])
    test = by_confidence(load(results, adapter(size))["rows"])
    lines = ["| Target accuracy | Automated | Accuracy on automated | Sent to people |", "|---|---|---|---|"]
    for target in TARGETS[key]:
        threshold = lowest_threshold_for(val, key, target)
        if threshold is None:
            lines.append(f"| {target:.0%} | not reachable on validation | | |")
            continue
        auto = [r for r in test if r["confidence"] >= threshold]
        share = len(auto) / len(test)
        lines.append(f"| {target:.0%} | {share:.1%} | {rate(auto, key):.1%} | {1 - share:.1%} |")
    return lines


def build(results: Path, n: int) -> str:
    sections = [
        f"## Same {n:,} test complaints", "", *comparison(results, n), "",
        f"## How much data you need (same {n:,} complaints)", "", *data_curve(results, n), "",
    ]
    for size in ROUTED:
        for key, label in (("product_correct", "product"), ("pair_correct", "product + issue")):
            sections += [f"## Confidence routing, n={size:,}, {label} (thresholds from validation, measured on test)", "",
                         *routing(results, size, key), ""]
    return "\n".join(sections)
