---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: finetune
datasets:
- PolyAI/banking77
library_name: transformers
pipeline_tag: text-generation
tags:
- intent-classification
- text-classification
- customer-support
- banking77
- lora
- peft
model-index:
- name: Qwen2.5-1.5B-Banking77-Intent
  results:
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      name: Banking77
      type: PolyAI/banking77
      split: test
    metrics:
    - type: accuracy
      value: 0.934
      name: Accuracy
---

# Qwen2.5-1.5B-Banking77-Intent

A fine-tune of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that classifies banking customer-support messages into the 77 intents of [Banking77](https://huggingface.co/datasets/PolyAI/banking77). Give it a customer message and it replies with a single intent label, such as `card_arrival`.

It was trained with LoRA in about 4 minutes on one GPU, for about $0.15 of compute. On the Banking77 test set it scores **93.4%**, ahead of Claude Sonnet 5.5 (83.4%) and Claude Haiku 5.5 (76.1%) when they're given the full label list in the prompt, at about **600× lower cost per request** than Sonnet.

This repo contains two versions of the same model:

| Path | What it is | Use it when |
|---|---|---|
| `/` (repo root) | Merged model: the LoRA weights folded into Qwen2.5-1.5B-Instruct | You want a standard model that loads anywhere |
| `lora/` | The LoRA adapter on its own (~18M parameters) | You already serve Qwen2.5-1.5B-Instruct and want to add this task on top, for example with vLLM multi-LoRA |

Code, training scripts and the evaluation harness: **[github.com/Johnson-f/tinytriage](https://github.com/Johnson-f/tinytriage)**

## Results

Banking77 test set, 3,076 messages, none of which appear in training.

| Model | Accuracy | Invalid answers | Avg input tokens | p50 latency | Cost / 1k requests |
|---|---|---|---|---|---|
| **This model** | **93.4%** | 0.1% | 37 | 0.51s | **$0.0036** |
| Claude Sonnet 5.5 | 83.4% | 2.1% | 818 | 1.48s | $2.15 |
| Claude Haiku 5.5 | 76.1% | 1.8% | 818 | 1.54s | $0.11 |
| Qwen2.5-1.5B-Instruct, not fine-tuned | 44.6% | 4.6% | 431 | 0.51s | $0.0044 |

- The Claude models were called through the Perplexity Agent API with the 77 labels listed in the prompt, reasoning off and no tools. Their costs are as reported by the API.
- This model was served with vLLM on one NVIDIA L4 at 32 concurrent requests. Its cost is GPU time divided by requests and excludes idle time and cold starts.
- "Invalid" means the reply wasn't one of the 77 labels.

### Confidence routing

The model's probability for its answer is a useful confidence score. Thresholds were chosen on a 500-message validation set and measured on the test set:

| Target accuracy | Handled automatically | Accuracy on those | Sent to a person |
|---|---|---|---|
| 98% | 89.0% | 97.8% | 11.0% |
| **99%** | **81.0%** | **98.8%** | 19.0% |
| 99.5% | 71.6% | 99.1% | 28.4% |

## Usage

**Use this exact system prompt.** The model was trained on it, and the labels aren't listed in it. A different prompt will lower accuracy.

```
Classify the banking customer message into its intent label.
```

Use greedy decoding (`do_sample=False` or `temperature=0`). Labels are at most about 15 tokens long, so `max_new_tokens=20` is enough.

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto")

messages = [
    {"role": "system", "content": "Classify the banking customer message into its intent label."},
    {"role": "user", "content": "My card hasn't came in the mail yet?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# card_arrival
```

### PEFT (adapter only)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype="auto")
model = PeftModel.from_pretrained(base, "johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent", subfolder="lora")
```

### vLLM (OpenAI-compatible server)

```bash
vllm serve johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent --served-model-name banking77
```

```python
import math
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="banking77",
    messages=[
        {"role": "system", "content": "Classify the banking customer message into its intent label."},
        {"role": "user", "content": "Why hasn't my transfer gone through yet?"},
    ],
    temperature=0,
    max_tokens=20,
    logprobs=True,
)
label = resp.choices[0].message.content
confidence = math.exp(sum(t.logprob for t in resp.choices[0].logprobs.content))
print(label, round(confidence, 3))  # send to a person if confidence is below your threshold
```

## Labels

<details>
<summary>All 77 labels</summary>

`Refund_not_showing_up`, `activate_my_card`, `age_limit`, `apple_pay_or_google_pay`, `atm_support`, `automatic_top_up`, `balance_not_updated_after_bank_transfer`, `balance_not_updated_after_cheque_or_cash_deposit`, `beneficiary_not_allowed`, `cancel_transfer`, `card_about_to_expire`, `card_acceptance`, `card_arrival`, `card_delivery_estimate`, `card_linking`, `card_not_working`, `card_payment_fee_charged`, `card_payment_not_recognised`, `card_payment_wrong_exchange_rate`, `card_swallowed`, `cash_withdrawal_charge`, `cash_withdrawal_not_recognised`, `change_pin`, `compromised_card`, `contactless_not_working`, `country_support`, `declined_card_payment`, `declined_cash_withdrawal`, `declined_transfer`, `direct_debit_payment_not_recognised`, `disposable_card_limits`, `edit_personal_details`, `exchange_charge`, `exchange_rate`, `exchange_via_app`, `extra_charge_on_statement`, `failed_transfer`, `fiat_currency_support`, `get_disposable_virtual_card`, `get_physical_card`, `getting_spare_card`, `getting_virtual_card`, `lost_or_stolen_card`, `lost_or_stolen_phone`, `order_physical_card`, `passcode_forgotten`, `pending_card_payment`, `pending_cash_withdrawal`, `pending_top_up`, `pending_transfer`, `pin_blocked`, `receiving_money`, `request_refund`, `reverted_card_payment?`, `supported_cards_and_currencies`, `terminate_account`, `top_up_by_bank_transfer_charge`, `top_up_by_card_charge`, `top_up_by_cash_or_cheque`, `top_up_failed`, `top_up_limits`, `top_up_reverted`, `topping_up_by_card`, `transaction_charged_twice`, `transfer_fee_charged`, `transfer_into_account`, `transfer_not_received_by_recipient`, `transfer_timing`, `unable_to_verify_identity`, `verify_my_identity`, `verify_source_of_funds`, `verify_top_up`, `virtual_card_not_working`, `visa_or_mastercard`, `why_verify_identity`, `wrong_amount_of_cash_received`, `wrong_exchange_rate_for_cash_withdrawal`

The capital `R` in `Refund_not_showing_up` and the `?` in `reverted_card_payment?` are part of the original dataset's labels.

</details>

## Training

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Method | Supervised fine-tuning with LoRA (TRL + PEFT), loss on the label only |
| LoRA | r=16, alpha=32, dropout=0.05, all linear layers |
| Optimizer | AdamW, learning rate 2e-4, cosine schedule, 20 warm-up steps |
| Batch size / epochs | 32 / 2 (594 steps) |
| Precision | bf16 |
| Data | 9,493 training and 500 validation messages from the Banking77 training split |
| Hardware | 1× NVIDIA L40S on Modal, 4 minutes 16 seconds |

Validation loss fell from 0.126 to 0.057 over training, with no sign of overfitting.

## Limitations

- **Banking only, and only these 77 intents.** The model will force any message into one of them, including messages that aren't about banking at all.
- **Clean benchmark data.** Banking77 messages are short and have one intent each. Real support tickets are longer, messier and sometimes have several intents, so expect lower accuracy on them without further fine-tuning on your own data.
- **Some labels overlap.** Most errors fall between near-identical intents (`exchange_rate` vs `exchange_charge`, `pending_transfer` vs `transfer_not_received_by_recipient`), where the dataset's own labels are inconsistent. The confidence score is low on many of these, so routing them to people works well.
- **English only.**
- **Not for automated financial decisions** without human review.

## License and citation

The model is released under Apache 2.0, the license of the base model. The training data is Banking77, released under CC BY 4.0:

```bibtex
@inproceedings{Casanueva2020,
  author    = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
  title     = {Efficient Intent Detection with Dual Sentence Encoders},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.04807}
}
```
