import random
from collections import Counter

from datasets import load_dataset

ds = load_dataset("mteb/banking77")
train, test = ds["train"], ds["test"]

print(f"train: {len(train)} examples, test: {len(test)} examples\n")

print("=== 30 random training examples ===")
random.seed(42)
for row in random.sample(list(train), 30):
    print(f"[{row['label_text']}]  {row['text']}")

train_counts = Counter(train["label_text"])
test_counts = Counter(test["label_text"])
ranked = train_counts.most_common()

print(f"\n=== {len(train_counts)} categories ===")
print("Most common in train:")
for label, n in ranked[:5]:
    print(f"  {n:4d}  {label}")
print("Least common in train:")
for label, n in ranked[-5:]:
    print(f"  {n:4d}  {label}")

print(f"\nTest set per category: min {min(test_counts.values())}, max {max(test_counts.values())}")
