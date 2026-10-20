# tinytriage

**A 1.5B open model, fine-tuned in 4 minutes for about $0.15, beats both Claude Sonnet 5.5 and Claude Haiku 5.5 at support-ticket triage. It scores 93.4% vs Sonnet's 83.4%, costs about 600× less per request than Sonnet and 29× less than Haiku, and responds about 3× faster. With confidence routing, it handles 81% of tickets automatically at 98.8% accuracy.**

**Model on Hugging Face:** [johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent](https://huggingface.co/johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent) (merged model plus the LoRA adapter in `lora/`)

This repo is the full pipeline behind that result: data preparation, LoRA fine-tuning on [Modal](https://modal.com), serving with [vLLM](https://github.com/vllm-project/vllm), an evaluation harness that scores any OpenAI-compatible endpoint, and a confidence-routing analysis.

## Results

**Task:** classify a banking customer's message into one of 77 intents ([Banking77](https://huggingface.co/datasets/mteb/banking77)).
**Test set:** 3,076 held-out messages that never appear in training.

| Model | Accuracy | Invalid answers | Avg input tokens | p50 latency | p95 latency | Cost / 1k requests |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B + LoRA (this repo)** | **93.4%** | 0.1% | 37 | 0.51s | 0.59s | **$0.0036** |
| Claude Sonnet 5.5 | 83.4% | 2.1% | 818 | 1.48s | 2.17s | $2.1528 |
| Claude Haiku 5.5 | 76.1% | 1.8% | 818 | 1.54s | 2.03s | $0.1055 |
| Qwen2.5-1.5B, not fine-tuned | 44.6% | 4.6% | 431 | 0.51s | 0.68s | $0.0044 |

- Claude models were called through the [Perplexity Agent API](https://docs.perplexity.ai/docs/agent-api/models) with the full list of 77 labels in the prompt, with reasoning turned off and no web-search tools. Their costs are the per-request costs Perplexity reported.
- The Qwen models were self-hosted on one NVIDIA L4 ($0.80/hr on Modal). Their cost is GPU time during the run, divided by the number of requests.
- "Invalid" means the answer wasn't one of the 77 labels. For Sonnet, these were mostly explanations of why a message was ambiguous instead of a label. Sonnet sometimes even mentioned the Banking77 dataset by name.

### Confidence routing: automate the sure cases, send the rest to people

Each answer's confidence is the probability the model assigns to the whole label: the product of its token probabilities, taken from vLLM's logprobs. Thresholds were **chosen on a separate 500-message validation set** and **measured on the test set**, so these numbers aren't tuned to the test data.

| Target accuracy | Automated | Accuracy on automated tickets | Sent to people |
|---|---|---|---|
| 98% | 89.0% | 97.8% | 11.0% |
| **99%** | **81.0%** | **98.8%** | 19.0% |
| 99.5% | 71.6% | 99.1% | 28.4% |
| 99.9% | 53.6% | 99.6% | 46.4% |

Each target is missed slightly on test because a 500-example validation set gives noisy thresholds. With real ticket data, use a larger validation set.

### Why the fine-tuned model is cheaper and faster

1. **Smaller model:** 1.5B parameters on an entry-level GPU.
2. **About 22× shorter prompts:** the model learned the 77 labels during training, so requests don't include the label list (37 input tokens vs 818).
3. **No hidden reasoning:** it outputs the label directly in about 5 tokens.

### Where it's still wrong

Most of the 203 test errors are in categories that overlap in meaning, where the dataset's own labels are inconsistent:

| Message | Dataset label | Model's answer |
|---|---|---|
| "whats your exchange rate" | `exchange_charge` | `exchange_rate` |
| "What all currencies can be exchanged?" | `exchange_via_app` | `fiat_currency_support` |
| "Do you offer additional currency options?" | `fiat_currency_support` | `exchange_via_app` |
| "Why hasn't my transfer gone through yet?" | `pending_transfer` | `transfer_not_received_by_recipient` |

In the last example, the model's confidence was 57.6%. That's below every threshold above, so the ticket would go to a person.

## How it works

```
                 Banking77 (train 9,493 / val 500 / test 3,076)
                                   │
              data/prepare.py ─────┤  prompt/completion JSONL, leak check
                                   │
   finetune/train.py ──► Modal L40S: LoRA SFT (TRL + PEFT) ──► adapter in Modal Volume
                                   │
   serve/serve.py ─────► Modal L4: vLLM, one server, two models:
                                   │   "base"      = Qwen2.5-1.5B-Instruct
                                   │   "banking77" = base + LoRA adapter (multi-LoRA)
                                   ▼
   eval/run_eval.py ───► scores ANY OpenAI-compatible endpoint the same way:
                         vLLM (chat API), Perplexity Agent API (responses API)
                                   │
              eval/report.py ──────┤  comparison table
              eval/coverage.py ────┘  confidence-routing table
```

### Fine-tuning setup

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | Supervised fine-tuning with LoRA, loss on the completion (the label) only |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, all linear layers (~18M trainable params, about 1.2% of the model) |
| Optimizer | AdamW, learning rate `2e-4`, cosine schedule, 20 warm-up steps |
| Batch / epochs | 32 / 2 (594 steps) |
| Precision | bf16 |
| Hardware | 1× NVIDIA L40S on Modal, 4m16s of training |

Training loss fell from 2.29 to about 0.02, and validation loss from 0.126 to 0.057 with no sign of overfitting.

The training prompt is a single sentence with no label list: *"Classify the banking customer message into its intent label."* This short prompt is a large part of the cost saving.

## Repository layout

```
data/
  explore.py      read random examples and the category distribution
  prepare.py      split train/val/test, write JSONL, check test/train overlap
finetune/
  train.py        Modal function: LoRA fine-tuning, saves adapter to a Modal Volume
  publish.py      merges the adapter, sanity-checks it and uploads both to Hugging Face
  MODEL_CARD.md   the Hugging Face model card
serve/
  serve.py        Modal web server: vLLM with the base model and LoRA adapter
eval/
  run_eval.py     evaluation harness (accuracy, invalid answers, latency, tokens, cost, confidence)
  report.py       builds the comparison table from results/*.json
  coverage.py     confidence-routing analysis (thresholds from val, measured on test)
hello.py          GPU check: runs nvidia-smi on a Modal T4
```

## Reproduce it

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- A [Modal](https://modal.com) account. The free Starter plan includes $30/month of compute, which covers everything here several times over.
- Optional, for the frontier-model baseline: a [Perplexity API](https://docs.perplexity.ai) key with a few dollars of credit, or any other OpenAI-compatible provider.

### 1. Set up

```bash
uv sync
uv run modal setup          # log in via the browser
uv run modal run hello.py   # should print a Tesla T4
```

### 2. Prepare the data

```bash
uv run python data/explore.py   # optional: read examples first
uv run python data/prepare.py   # writes data/{train,val,test}.jsonl and data/labels.json
```

### 3. Fine-tune (~5 minutes, ~$0.15 of GPU time)

```bash
uv run modal run finetune/train.py            # default 2 epochs
uv run modal run finetune/train.py --epochs 3
```

The adapter is saved to the Modal Volume `banking77-adapters`, and the training log to `results/train_log.json`.

### 4. Serve

```bash
export VLLM_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
uv run modal serve serve/serve.py
```

`modal serve` prints a URL like `https://<workspace>--banking77-serve-serve-dev.<region>.modal.run`. The first start takes a few minutes while the image builds and the model loads. The server stops when you press Ctrl-C.

### 5. Evaluate

In a second terminal, with the same `VLLM_API_KEY`:

```bash
URL=https://<your-url>.modal.run/v1

# not fine-tuned (needs the label list in the prompt)
uv run python eval/run_eval.py --name qwen1.5b-base --base-url $URL --model base \
  --api-key-env VLLM_API_KEY --prompt full --concurrency 32 --gpu-hourly 0.80

# fine-tuned (short prompt)
uv run python eval/run_eval.py --name qwen1.5b-finetuned --base-url $URL --model banking77 \
  --api-key-env VLLM_API_KEY --prompt short --concurrency 32 --gpu-hourly 0.80

# frontier baseline via the Perplexity Agent API
uv run python eval/run_eval.py --name frontier-haiku-5.5 --api responses --reasoning-effort none \
  --base-url https://api.perplexity.ai/v1 --model anthropic/claude-haiku-5-5 \
  --api-key-env PERPLEXITY_API_KEY --prompt full --concurrency 16

uv run python eval/report.py
```

### 6. Confidence routing

```bash
for SPLIT in val test; do
  uv run python eval/run_eval.py --name qwen1.5b-finetuned-conf-$SPLIT --split $SPLIT --logprobs \
    --base-url $URL --model banking77 --api-key-env VLLM_API_KEY --prompt short --concurrency 32
done
uv run python eval/coverage.py --val qwen1.5b-finetuned-conf-val --test qwen1.5b-finetuned-conf-test
```

### 7. Publish to Hugging Face (optional)

Put a Hugging Face token with write access in `.env` as `HF_TOKEN=...`, then:

```bash
uv run --env-file .env modal run finetune/publish.py --repo-id <user>/<model-name>            # merge + sanity check only
uv run --env-file .env modal run finetune/publish.py --repo-id <user>/<model-name> --upload   # also upload
```

The script refuses to upload if the merged model scores below 80% on a 20-message sanity check.

### Harness options

| Flag | Purpose |
|---|---|
| `--api chat \| responses` | Chat Completions (vLLM, most providers) or the Responses API (Perplexity Agent API) |
| `--prompt short \| full` | Short instruction (fine-tuned models) or instruction plus all 77 labels (anything else) |
| `--split test \| val` | Which dataset split to score |
| `--logprobs` | Record each answer's confidence (chat API) |
| `--reasoning-effort` | Responses API only. Set to `none` for classification (see lessons below). |
| `--price-in`, `--price-out` | USD per 1M tokens, if the provider doesn't report cost |
| `--gpu-hourly` | USD per GPU-hour, for self-hosted cost per request |
| `--limit N` | Score only the first N examples (for quick tests) |

## Lessons learned

- **Read raw outputs before trusting a score.** The first Haiku score was 72.8%, and the real figure is 76.1%. The difference came from two bugs in the evaluation code: lowercasing broke the one capitalized label (`Refund_not_showing_up`), and a 20-token output cap cut long labels in half. Four categories at exactly 0% were the giveaway.
- **Reasoning defaults cost money and can break a classifier.** By default, Haiku used its whole output budget on hidden reasoning tokens for ambiguous messages and returned an empty answer. Setting `reasoning.effort = none` fixed it, and it also means paying for about 7 output tokens instead of 100+.
- **Slim container images can break vLLM.** vLLM's FlashInfer sampler compiles GPU kernels at startup and needs `nvcc`, which `debian_slim` doesn't include. Setting `VLLM_USE_FLASHINFER_SAMPLER=0` avoids it; greedy decoding doesn't need that sampler.
- **Never set thresholds on the test set.** Thresholds chosen on validation land slightly below target on test. That gap is the honest number to report.

## Caveats

- **Self-hosted cost** is GPU time during a run with 32 concurrent requests. It excludes idle time and cold starts (about 1–4 minutes). vLLM reported room for about 290 concurrent requests, so heavier traffic would lower the cost per request. Low traffic would raise it unless the server scales to zero.
- **Latency** includes the network round-trip to Modal's `us-east` region.
- **Banking77 is clean, short, single-intent data.** Real support tickets are longer, messier and sometimes cover several issues, so expect lower accuracy on them until the model is trained on that data.
- **Claude models ran with reasoning off**, which is how you'd normally run a high-volume classifier. With reasoning on they may score higher, at a much higher cost.

## Cost of this experiment

| Item | Approx. cost |
|---|---|
| LoRA training (L40S) | ~$0.15 of GPU time, plus a one-time image build and model download |
| Serving and evaluation (L4) | < $1 |
| Claude Haiku 5.5, full test set | ~$0.32 |
| Claude Sonnet 5.5, full test set | ~$6.62 |
| **Total** | **~$8–9** |

## Credits

- Dataset: Banking77, from Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), via the [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) mirror.
- Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- Built with [Modal](https://modal.com), [vLLM](https://github.com/vllm-project/vllm), [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft).
