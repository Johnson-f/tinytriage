"""Build the CFPB complaint-routing dataset: dedupe, time split, balance by product."""

import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(os.environ.get("CFPB_DATA", "/data"))
RAW = DATA / "raw" / "cfpb"
OUT = DATA / "tasks" / "cfpb"

TRAIN_END = "2025-10-15"
VAL_END = "2025-10-31"
TEST_START = "2025-11-01"
MAX_WORDS = 500
TEST_PER_PRODUCT = 300
VAL_PER_PRODUCT = 100
TRAIN_POOL_SIZE = 20_000
SEED = 42

csv.field_size_limit(10**8)


def normalize(text: str) -> str:
    # Credit-repair templates are refiled with different redacted names and numbers,
    # so redaction marks and digits are collapsed before comparing texts.
    text = text.lower()
    text = re.sub(r"x{2,}", "x", text)
    text = re.sub(r"\d", "0", text)
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str) -> str:
    words = text.split()
    return " ".join(words[:MAX_WORDS]) + (" …" if len(words) > MAX_WORDS else "")


def read_complaints() -> list[dict]:
    rows = []
    for path in sorted(RAW.glob("*/*.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                text = (row["Consumer complaint narrative"] or "").strip()
                if not text:
                    continue
                rows.append({
                    "id": row["Complaint ID"],
                    "date": row["Date received"],
                    "product": row["Product"],
                    "issue": row["Issue"],
                    "text": text,
                })
    return sorted(rows, key=lambda r: (r["date"], r["id"]))


def dedupe(rows: list[dict]) -> list[dict]:
    seen, kept = set(), []
    for row in rows:
        key = hashlib.md5(normalize(row["text"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def cap_per_product(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    by_product = defaultdict(list)
    for row in rows:
        by_product[row["product"]].append(row)
    picked = []
    for items in by_product.values():
        rng.shuffle(items)
        picked.extend(items[:cap])
    rng.shuffle(picked)
    return picked


def round_robin(rows: list[dict], size: int, rng: random.Random) -> list[dict]:
    # One ordering across products, so the first N rows are as balanced as possible
    # and every smaller training set is a prefix of the larger ones.
    queues = defaultdict(list)
    for row in rows:
        queues[row["product"]].append(row)
    for items in queues.values():
        rng.shuffle(items)
    order, products = [], sorted(queues)
    while len(order) < size and any(queues.values()):
        for product in products:
            if queues[product] and len(order) < size:
                order.append(queues[product].pop())
    return order


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps({**row, "text": truncate(row["text"])}) + "\n")


def main() -> dict:
    rng = random.Random(SEED)
    raw = read_complaints()
    unique = dedupe(raw)

    train_all = [r for r in unique if r["date"] <= TRAIN_END]
    val_all = [r for r in unique if TRAIN_END < r["date"] <= VAL_END]
    test_all = [r for r in unique if r["date"] >= TEST_START]

    known_pairs = {(r["product"], r["issue"]) for r in train_all}
    val_known = [r for r in val_all if (r["product"], r["issue"]) in known_pairs]
    test_known = [r for r in test_all if (r["product"], r["issue"]) in known_pairs]

    train_pool = round_robin(train_all, TRAIN_POOL_SIZE, rng)
    val = cap_per_product(val_known, VAL_PER_PRODUCT, rng)
    test = cap_per_product(test_known, TEST_PER_PRODUCT, rng)

    labels = defaultdict(set)
    for product, issue in known_pairs:
        labels[product].add(issue)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train_pool.jsonl", train_pool)
    write_jsonl(OUT / "val.jsonl", val)
    write_jsonl(OUT / "test.jsonl", test)
    (OUT / "labels.json").write_text(json.dumps({p: sorted(i) for p, i in sorted(labels.items())}, indent=2))

    stats = {
        "narratives_read": len(raw),
        "after_dedupe": len(unique),
        "train_period_unique": len(train_all),
        "val_period_unique": len(val_all),
        "test_period_unique": len(test_all),
        "test_dropped_unseen_label": len(test_all) - len(test_known),
        "products": len(labels),
        "product_issue_pairs": len(known_pairs),
        "train_pool": len(train_pool),
        "val": len(val),
        "test": len(test),
        "test_per_product": dict(Counter(r["product"] for r in test)),
        "train_pool_first_1000_per_product": dict(Counter(r["product"] for r in train_pool[:1000])),
        "words_truncated_share": sum(len(r["text"].split()) > MAX_WORDS for r in test) / max(len(test), 1),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
