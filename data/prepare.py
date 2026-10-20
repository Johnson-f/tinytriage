import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent
VAL_SIZE = 500
FINETUNE_SYSTEM_PROMPT = "Classify the banking customer message into its intent label."


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def to_training_example(row: dict) -> dict:
    # TRL's prompt/completion format: loss is computed on the completion only,
    # so the model learns to emit labels rather than to reproduce customer messages.
    return {
        "prompt": [
            {"role": "system", "content": FINETUNE_SYSTEM_PROMPT},
            {"role": "user", "content": row["text"]},
        ],
        "completion": [{"role": "assistant", "content": row["label_text"]}],
    }


def main() -> None:
    ds = load_dataset("mteb/banking77")
    train_rows = list(ds["train"])
    test_rows = list(ds["test"])

    random.seed(42)
    random.shuffle(train_rows)
    val_rows, train_rows = train_rows[:VAL_SIZE], train_rows[VAL_SIZE:]

    labels = sorted({row["label_text"] for row in ds["train"]})
    (DATA_DIR / "labels.json").write_text(json.dumps(labels, indent=2))

    write_jsonl(DATA_DIR / "train.jsonl", [to_training_example(r) for r in train_rows])
    write_jsonl(DATA_DIR / "val.jsonl", [to_training_example(r) for r in val_rows])
    write_jsonl(DATA_DIR / "test.jsonl", [{"text": r["text"], "label": r["label_text"]} for r in test_rows])

    test_texts = {r["text"] for r in test_rows}
    leaked = sum(r["text"] in test_texts for r in train_rows + val_rows)

    print(f"labels: {len(labels)}")
    print(f"train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")
    print(f"test messages also present in train/val: {leaked}")


if __name__ == "__main__":
    main()
