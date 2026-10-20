import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
TARGET_ACCURACIES = [0.98, 0.99, 0.995, 0.999]
COVERAGE_LEVELS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]


def load_rows(name: str) -> list[dict]:
    rows = json.loads((RESULTS_DIR / f"{name}.json").read_text())["rows"]
    return sorted(rows, key=lambda r: r["confidence"], reverse=True)


def accepted(rows: list[dict], threshold: float) -> list[dict]:
    return [r for r in rows if r["confidence"] >= threshold]


def accuracy(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def lowest_threshold_for(rows: list[dict], target: float) -> float | None:
    # Walk from most to least confident; the last point still meeting the target
    # gives the most automation at that accuracy.
    best, correct = None, 0
    for i, r in enumerate(rows, start=1):
        correct += r["correct"]
        if correct / i >= target:
            best = r["confidence"]
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Accuracy vs automation rate from answer confidence.")
    p.add_argument("--val", required=True, help="results name for the validation run (picks thresholds)")
    p.add_argument("--test", required=True, help="results name for the test run (reports honestly)")
    args = p.parse_args()

    val, test = load_rows(args.val), load_rows(args.test)
    lines = [
        "## Thresholds chosen on validation, measured on test",
        "",
        "| Target accuracy | Threshold | Test: automated | Test: accuracy on automated | Sent to humans |",
        "|---|---|---|---|---|",
    ]
    for target in TARGET_ACCURACIES:
        threshold = lowest_threshold_for(val, target)
        if threshold is None:
            lines.append(f"| {target:.1%} | not reachable on val | | | |")
            continue
        auto = accepted(test, threshold)
        lines.append(
            f"| {target:.1%} | {threshold:.4f} | {len(auto) / len(test):.1%} "
            f"| {accuracy(auto):.2%} | {1 - len(auto) / len(test):.1%} |"
        )

    lines += [
        "",
        "## Test-set curve: automate the most confident X%",
        "",
        "| Automated | Accuracy on automated | Errors per 1,000 automated |",
        "|---|---|---|",
    ]
    for level in COVERAGE_LEVELS:
        top = test[: round(level * len(test))]
        acc = accuracy(top)
        lines.append(f"| {level:.0%} | {acc:.2%} | {(1 - acc) * 1000:.1f} |")

    report = "\n".join(lines)
    (RESULTS_DIR / "coverage.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
