import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_summaries() -> list[dict]:
    summaries = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "summary" in data and data["summary"].get("split", "test") == "test":
            summaries.append(data["summary"])
    return sorted(summaries, key=lambda s: s["accuracy"], reverse=True)


def fmt_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def main() -> None:
    summaries = load_summaries()
    lines = [
        "| Model | Prompt | Accuracy | Invalid | Avg input tokens | p50 latency | p95 latency | Cost / 1k requests |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['prompt']} | {s['accuracy']:.1%} | {s['invalid_label_rate']:.1%} "
            f"| {s['avg_input_tokens']:.0f} | {s['latency_p50_s']:.2f}s | {s['latency_p95_s']:.2f}s "
            f"| {fmt_cost(s['cost_per_1k_requests_usd'])} |"
        )
    table = "\n".join(lines)
    (RESULTS_DIR / "table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
