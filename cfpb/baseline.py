"""Non-LLM baseline: TF-IDF + logistic regression, at several training-set sizes."""

import json
import os
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = Path(os.environ.get("CFPB_DATA", "/data")) / "tasks" / "cfpb"
RESULTS = DATA / "results"
SIZES = [100, 500, 1000, 5000, 20000]
# Frontier models were scored on the first 1,000 test rows; report that subset too for a like-for-like comparison.
FRONTIER_SAMPLE = 1000


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).open()]


def pair(row: dict) -> str:
    return f"{row['product']} || {row['issue']}"


def per_product_accuracy(rows: list[dict], preds: list[str]) -> dict[str, float]:
    hits = defaultdict(list)
    for row, pred in zip(rows, preds):
        hits[row["product"]].append(pred == pair(row))
    return {product: sum(h) / len(h) for product, h in hits.items()}


def main() -> list[dict]:
    pool, test = load("train_pool.jsonl"), load("test.jsonl")
    test_text = [r["text"] for r in test]
    results = []
    for size in SIZES:
        train = pool[:size]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if size >= 1000 else 1, sublinear_tf=True)
        x_train = vectorizer.fit_transform([r["text"] for r in train])
        model = LogisticRegression(max_iter=2000, C=5.0)
        model.fit(x_train, [pair(r) for r in train])
        preds = model.predict(vectorizer.transform(test_text))

        pair_acc = sum(p == pair(r) for p, r in zip(preds, test)) / len(test)
        product_acc = sum(p.split(" || ")[0] == r["product"] for p, r in zip(preds, test)) / len(test)
        by_product = per_product_accuracy(test, preds)
        sub_test, sub_preds = test[:FRONTIER_SAMPLE], preds[:FRONTIER_SAMPLE]
        results.append({
            "train_size": size,
            "pair_accuracy": pair_acc,
            "product_accuracy": product_acc,
            "macro_pair_accuracy": sum(by_product.values()) / len(by_product),
            "sample1000_pair_accuracy": sum(p == pair(r) for p, r in zip(sub_preds, sub_test)) / len(sub_test),
            "sample1000_product_accuracy": sum(p.split(" || ")[0] == r["product"] for p, r in zip(sub_preds, sub_test)) / len(sub_test),
        })
        print(f"n={size:6d}  product {product_acc:6.1%}  product+issue {pair_acc:6.1%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baseline_tfidf.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
