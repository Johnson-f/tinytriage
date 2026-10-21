---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
tags:
- text-classification
- complaint-routing
- customer-support
- finance
- cfpb
- lora
- peft
model-index:
- name: Qwen2.5-1.5B-CFPB-Complaint-Routing
  results:
  - task:
      type: text-classification
      name: Complaint routing (product and issue)
    dataset:
      name: CFPB Consumer Complaint Database (Nov–Dec 2025)
      type: cfpb-consumer-complaints
      split: test
    metrics:
    - type: accuracy
      value: 0.778
      name: Product accuracy
    - type: accuracy
      value: 0.491
      name: Product and issue accuracy
---

# Qwen2.5-1.5B-CFPB-Complaint-Routing

A fine-tune of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that routes consumer financial complaints. Give it a complaint written by a customer and it replies with JSON naming the **product** (one of 11, such as `Credit card` or `Mortgage`) and the **issue** (one of 132 product–issue pairs, such as `Trouble using the card`).

It was trained with LoRA on 20,000 real complaints from the US [Consumer Financial Protection Bureau (CFPB) complaint database](https://www.consumerfinance.gov/data-research/consumer-complaints/). On 1,000 later complaints it **matches Claude Sonnet 5.5** (77.5% vs 75.6% on product, 50.0% vs 48.5% on product and issue) at about **1/390th of the cost per request**.

This repo contains two versions of the same model:

| Path | What it is | Use it when |
|---|---|---|
| `/` (repo root) | Merged model: the LoRA weights folded into Qwen2.5-1.5B-Instruct | You want a standard model that loads anywhere |
| `lora/` | The LoRA adapter on its own (~18M parameters) | You already serve Qwen2.5-1.5B-Instruct and want to add this task on top, for example with vLLM multi-LoRA |

Code, training scripts and the evaluation harness: **[github.com/Johnson-f/tinytriage](https://github.com/Johnson-f/tinytriage)**

## Results

The test complaints were filed **after** every training complaint (time-based split), the way a deployed model meets new data. Each product has 300 test complaints, so no single product dominates.

**Same 1,000 test complaints for every model:**

| Model | Product | Product + issue | Avg input tokens | p50 latency | Cost / 1k requests |
|---|---|---|---|---|---|
| **This model** | **77.5%** | **50.0%** | 295 | 0.99s | **$0.0072** |
| Claude Sonnet 5.5 | 75.6% | 48.5% | 2,533 | 1.39s | $2.82 |
| Claude Haiku 5.5 | 75.2% | 46.1% | 2,533 | 2.34s | $0.46 |
| TF-IDF + logistic regression, same 20,000 examples | 69.9% | 47.1% | | | ~$0 |
| Qwen2.5-1.5B-Instruct, not fine-tuned | 47.4% | 5.5% | 1,626 | 0.96s | $0.0067 |

On all 3,300 test complaints, this model scores 77.8% on product and 49.1% on product and issue, and every answer is valid JSON with a real label.

- The Claude models were called through the Perplexity Agent API with every product and issue listed in the prompt and reasoning effort `high`. Their costs are as reported by the API.
- This model was served with vLLM on one NVIDIA L4 ($0.80/hr) at 32 concurrent requests. Its cost is GPU time divided by requests and excludes idle time and cold starts.
- The gap to Sonnet is within sampling noise for 1,000 complaints. The fair summary is "as accurate as Sonnet, far cheaper".

**Why product + issue is only about 50% for every model:** customers choose the product and issue themselves when they file, so the labels are noisy, and many issues overlap (the label set even contains both `Problem with a company's investigation into an existing problem` and `...existing issue`). Product accuracy is the more reliable number.

### Confidence routing

The model's probability for its whole answer works as a confidence score. Thresholds were chosen on 1,093 validation complaints and measured on the 3,300 test complaints:

| Routing by product: target accuracy | Handled automatically | Accuracy on those |
|---|---|---|
| 85% | 77.6% | 84.8% |
| **90%** | **57.8%** | **88.8%** |
| 95% | 24.8% | 94.7% |

| Product + issue: target accuracy | Handled automatically | Accuracy on those |
|---|---|---|
| 70% | 47.6% | 69.9% |
| 80% | 28.9% | 80.3% |
| 90% | 11.5% | 92.1% |

## Usage

**Use this exact system prompt.** The model was trained on it, and the labels aren't listed in it. A different prompt will lower accuracy.

```
Classify this consumer financial complaint. Reply with JSON: {"product": ..., "issue": ...}
```

Use greedy decoding (`do_sample=False` or `temperature=0`) and `max_new_tokens=80`.

### Transformers

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "johnsonfzzz/Qwen2.5-1.5B-CFPB-Complaint-Routing"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto")

messages = [
    {"role": "system", "content": 'Classify this consumer financial complaint. Reply with JSON: {"product": ..., "issue": ...}'},
    {"role": "user", "content": "I received a gift card for $100.00 from a friend. Its an American Express gift card. It wont spend in-person or online."},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
answer = json.loads(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
print(answer)
# {'product': 'Prepaid card', 'issue': 'Trouble using the card'}
```

### PEFT (adapter only)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype="auto")
model = PeftModel.from_pretrained(base, "johnsonfzzz/Qwen2.5-1.5B-CFPB-Complaint-Routing", subfolder="lora")
```

### vLLM (OpenAI-compatible server) with confidence

```bash
vllm serve johnsonfzzz/Qwen2.5-1.5B-CFPB-Complaint-Routing --served-model-name cfpb --max-model-len 4096
```

```python
import json
import math
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="cfpb",
    messages=[
        {"role": "system", "content": 'Classify this consumer financial complaint. Reply with JSON: {"product": ..., "issue": ...}'},
        {"role": "user", "content": "A debt collector keeps calling me about a loan I never took out."},
    ],
    temperature=0,
    max_tokens=80,
    logprobs=True,
)
answer = json.loads(resp.choices[0].message.content)
confidence = math.exp(sum(t.logprob for t in resp.choices[0].logprobs.content))
print(answer, round(confidence, 3))  # send to a person if confidence is below your threshold
```

## Labels

The 11 products are: Checking or savings account; Credit card; Credit reporting or other personal consumer reports; Debt collection; Debt or credit management; Money transfer, virtual currency, or money service; Mortgage; Payday loan, title loan, personal loan, or advance loan; Prepaid card; Student loan; Vehicle loan or lease. Issues are the CFPB's own issue names for each product, 132 product–issue pairs in all.

## Training

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Method | Supervised fine-tuning with LoRA (TRL + PEFT), loss on the JSON answer only |
| LoRA | r=16, alpha=32, dropout=0.05, all linear layers |
| Optimizer | AdamW, learning rate 2e-4, cosine schedule, 20 warm-up steps |
| Batch size / epochs | 16 / 2 (2,500 steps); the checkpoint with the lowest validation loss (step 2,250) was kept |
| Precision | bf16 |
| Max length | 1,024 tokens; complaints were cut to 500 words (8% were longer) |
| Hardware | 1× NVIDIA L40S on Modal, 82 minutes |

**Data.** Complaints with a written narrative from July to December 2025, deduplicated first: 49% of narratives were exact or near-exact copies (often the same template letter filed many times), which would otherwise leak between training and test.

| Split | Filed | Size |
|---|---|---|
| Train | Jul 1 – Oct 15, 2025 | 20,000, balanced across products |
| Validation | Oct 16 – Oct 31, 2025 | 1,093 |
| Test | Nov 1 – Dec 31, 2025 | 3,300 (300 per product) |

## Limitations

- **US consumer complaints, these 11 products and their CFPB issue names only.** The model forces every input into one of them.
- **Noisy labels.** Customers choose their own product and issue, so some "errors" are reasonable answers that disagree with the filer.
- **Balanced test set.** Real complaint volume is dominated by credit reporting, so accuracy on a natural mix will differ.
- **Time drift.** Trained on mid-to-late 2025. Products, companies and complaint templates change, so retrain on recent data.
- **English only.**
- **Not for automated decisions about consumers** without human review.

## License and data

The model is released under Apache 2.0, the license of the base model. Training data comes from the CFPB Consumer Complaint Database, published by the US government. Consumers consent to publishing their narratives, and the CFPB removes personal information before release (shown as `XXXX`).
