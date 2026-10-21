"""CFPB complaint routing: prompts, the JSON answer format and its parser, shared by training and evaluation."""

import json

SHORT_PROMPT = 'Classify this consumer financial complaint. Reply with JSON: {"product": ..., "issue": ...}'


def full_prompt(labels: dict[str, list[str]]) -> str:
    lines = [
        "You route consumer financial complaints. Choose the product and the issue from the list below, "
        "using the exact wording. The issue must belong to the chosen product.",
        'Reply with JSON only, in the form {"product": "...", "issue": "..."}.',
        "",
    ]
    for product, issues in labels.items():
        lines.append(f"Product: {product}")
        lines.extend(f"  - {issue}" for issue in issues)
    return "\n".join(lines)


def completion(row: dict) -> str:
    return json.dumps({"product": row["product"], "issue": row["issue"]})


def parse(raw: str) -> dict | None:
    # Models sometimes wrap JSON in code fences or add text around it.
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("product"), str) or not isinstance(value.get("issue"), str):
        return None
    return {"product": value["product"].strip(), "issue": value["issue"].strip()}
