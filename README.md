# tinytriage

**Small fine-tuned open models vs Claude on support and complaint triage, scored on the same data.** A 1.5B model fine-tuned with LoRA beats Claude Sonnet 5.5 on clean support intents and matches it on messy real-world financial complaints, at a few hundredths of a cent per request.

| Benchmark | Task | Fine-tuned Qwen2.5-1.5B | Claude Sonnet 5.5 | Claude Haiku 5.5 | Cost vs Sonnet |
|---|---|---|---|---|---|
| [Banking77](#benchmark-1-banking77) | 77 intents, short chat messages | **93.4%** | 83.4% | 76.1% | ~600× cheaper |
| [CFPB complaints](#benchmark-2-cfpb-consumer-complaints) | Product and issue as JSON, real complaints | **77.5%** product, **50.0%** both | 75.6%, 48.5% | 75.2%, 46.1% | ~390× cheaper |

**Models on Hugging Face** (each is a merged model plus the LoRA adapter in `lora/`):
- [johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent](https://huggingface.co/johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent)
- [johnsonfzzz/Qwen2.5-1.5B-CFPB-Complaint-Routing](https://huggingface.co/johnsonfzzz/Qwen2.5-1.5B-CFPB-Complaint-Routing)

This repo is the full pipeline behind those results: data preparation, LoRA fine-tuning on [Modal](https://modal.com), serving with [vLLM](https://github.com/vllm-project/vllm), an evaluation harness that scores any OpenAI-compatible endpoint, and a confidence-routing analysis.

## Benchmark 1: Banking77

**A 1.5B open model, fine-tuned in 4 minutes for about $0.15, beats both Claude Sonnet 5.5 and Claude Haiku 5.5 at support-ticket triage. It scores 93.4% vs Sonnet's 83.4%, costs about 600× less per request than Sonnet and 29× less than Haiku, and responds about 3× faster. With confidence routing, it handles 81% of tickets automatically at 98.8% accuracy.**

### Results

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

## Benchmark 2: CFPB consumer complaints

**Banking77 is clean. Real complaints aren't.** This benchmark uses complaints that US consumers filed with the [Consumer Financial Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/) in July–December 2025: long, angry, sometimes copy-pasted letters about credit cards, mortgages, debt collectors and more. The model reads a complaint and returns JSON:

```json
{"product": "Prepaid card", "issue": "Trouble using the card"}
```

There are 11 products and 132 product–issue pairs. Customers choose both labels themselves when they file, so the labels are noisy, which is why every model's product + issue score is around 50%.

### Results

**Same 1,000 test complaints for every model.** The test complaints were filed after every training complaint.

| Model | Product | Product + issue | Invalid | Avg input tokens | p50 latency | Cost / 1k requests |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B + LoRA, 20,000 examples** | **77.5%** | **50.0%** | 0.0% | 295 | 0.99s | **$0.0072** |
| Claude Sonnet 5.5 | 75.6% | 48.5% | 0.0% | 2,533 | 1.39s | $2.8243 |
| Claude Haiku 5.5 | 75.2% | 46.1% | 0.0% | 2,533 | 2.34s | $0.4622 |
| TF-IDF + logistic regression, 20,000 examples | 69.9% | 47.1% | 0.0% | | | ~$0 |
| Qwen2.5-1.5B, not fine-tuned | 47.4% | 5.5% | 28.1% | 1,626 | 0.96s | $0.0067 |

- On all 3,300 test complaints, the fine-tuned model scores 77.8% product and 49.1% product + issue, with no invalid answers.
- The lead over Sonnet is within sampling noise for 1,000 complaints. The honest claim is **as accurate as Sonnet at about 1/390th of the cost**.
- Claude models got every product and issue in the prompt, with reasoning effort `high` (see lessons below for why not `none`). The fine-tuned model gets a one-line prompt, so it uses 8.6× fewer input tokens.
- "Invalid" means the answer wasn't valid JSON or named a product–issue pair that doesn't exist.

### How much data you need

The same model and settings, trained on the first N complaints of a balanced pool, against a classic TF-IDF + logistic regression classifier trained on the same N. Same 1,000 test complaints.

| Training examples | Fine-tuned: product | Fine-tuned: product + issue | TF-IDF: product | TF-IDF: product + issue |
|---|---|---|---|---|
| 0 (base model) | 47.4% | 5.5% | | |
| 100 | 59.5% | 18.8% | 24.9% | 15.0% |
| 500 | 67.3% | 34.4% | 41.0% | 25.5% |
| 1,000 | 71.9% | 41.4% | 53.7% | 33.4% |
| 5,000 | 73.3% | 44.4% | 65.0% | 42.6% |
| 20,000 | **77.5%** | **50.0%** | 69.9% | 47.1% |

- Fine-tuning wins at every size, by the most at 500–1,000 examples, where it already knows what a "mortgage" or a "debt collector" is.
- At 20,000 examples, plain TF-IDF beats Haiku on product + issue. Always try the cheap baseline before calling an LLM.

### Confidence routing

Confidence is the probability the model gives its whole JSON answer. Thresholds were **chosen on 1,093 validation complaints** (Oct 16–31) and **measured on the 3,300 test complaints**.

**Routing by product** (which team gets the complaint):

| Target accuracy | Automated | Accuracy on automated | Sent to people |
|---|---|---|---|
| 85% | 77.6% | 84.8% | 22.4% |
| **90%** | **57.8%** | **88.8%** | 42.2% |
| 95% | 24.8% | 94.7% | 75.2% |

**Product + issue:**

| Target accuracy | Automated | Accuracy on automated | Sent to people |
|---|---|---|---|
| 60% | 73.0% | 60.2% | 27.0% |
| 70% | 47.6% | 69.9% | 52.4% |
| 80% | 28.9% | 80.3% | 71.1% |
| 90% | 11.5% | 92.1% | 88.5% |

More data helps routing, not only accuracy: at the 80% product + issue target, the 5,000-example model automates 20.7% of complaints, while the 20,000-example model automates 28.9%.

### Where it's still wrong

| Complaint (shortened) | Filed as | Model's answer | Confidence |
|---|---|---|---|
| "Debt collector called my number, which is on the Do Not Call Registry. They called for my brother…" | Attempts to collect debt not owed | Communication tactics | 0.34 |
| "I was notified through the credit reporting agencies that a credit card was opened in my name…" | Other features, terms, or problems | Getting a credit card | 0.85 |
| "Solar panel loan for company that went bankrupt and panels do nothing…" | Getting the loan | Problem when making payments | 0.12 |

The first is arguably the model's answer, not the filer's. The low-confidence cases go to a person under any threshold above.

### How the dataset was built

```
 CFPB exports, Jul–Dec 2025 (526,866 complaints with a narrative)
        │
        ├─ dedupe: lowercase, collapse XXXX redactions and digits ──► 271,266 unique (49% were copies)
        ├─ truncate: narratives over 500 words are cut (8% of test)
        ├─ time split:  train Jul 1–Oct 15  │  val Oct 16–31  │  test Nov 1–Dec 31
        └─ balance:     20,000 train pool (round-robin over 11 products, nested prefixes for N=100…20,000)
                        1,093 val (≤100 per product) · 3,300 test (300 per product)
```

Deduplication matters: credit-repair template letters are refiled thousands of times with only the redacted names changed. Without it, the same letter lands in training and test, and every score is inflated.

### Fine-tuning setup

Same LoRA settings as Banking77 (`r=16`, `alpha=32`, all linear layers, learning rate `2e-4`, cosine), with batch size 16 and max length 1,024. Small training sets get more epochs so every size takes about 200 optimizer steps (capped at 20 epochs), and the checkpoint with the lowest validation loss is kept. The 20,000-example model trained for 2 epochs (2,500 steps) on one L40S in 82 minutes; the best checkpoint was step 2,250 (validation loss 0.080).

## How it works (Banking77)

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
  train.py        Modal function: LoRA fine-tuning (--task banking77 | cfpb), saves adapters to a Modal Volume
  publish.py      merges an adapter, sanity-checks it and uploads both to Hugging Face
  MODEL_CARD.md       Hugging Face model card, Banking77
  MODEL_CARD_CFPB.md  Hugging Face model card, CFPB
cfpb/
  modal_app.py    Modal: download the CFPB exports, build the dataset, run the TF-IDF baseline
  prepare.py      dedupe, time split, balance by product
  baseline.py     TF-IDF + logistic regression at each training size
  task.py         prompts, JSON answer format and parser (shared by training and evaluation)
  modal_eval.py   Modal: evaluate fine-tuned models (vLLM inside the job) and Claude, build the report
  evaluate.py     scoring: product / product + issue accuracy, invalid answers, latency, tokens, cost, confidence
  report.py       comparison, data-size curve and confidence-routing tables
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

The script refuses to upload if the merged model scores below 80% on a 20-message sanity check. For the CFPB model add `--task cfpb`; it checks product accuracy on 50 test complaints and refuses below 60%.

### 8. CFPB benchmark

Everything runs on Modal; the data lives in the Modal Volume `tinytriage-data`, and only small summary files come back to `results/cfpb/`.

```bash
# download Jul–Dec 2025 exports (~2 GB), build the dataset, run the TF-IDF baseline
uv run modal run cfpb/modal_app.py --step all

# fine-tune at each training size (each size runs as its own L40S job)
uv run modal run finetune/train.py --task cfpb --sizes 100,500,1000,5000,20000

# evaluate: base model + every CFPB adapter on the test set, then the validation set for routing
uv run modal run cfpb/modal_eval.py --step finetuned
uv run modal run cfpb/modal_eval.py --step finetuned --only qwen2.5-1.5b-cfpb-n5000,qwen2.5-1.5b-cfpb-n20000 \
  --no-include-base --split val

# Claude via the Perplexity Agent API, first 1,000 test complaints
PERPLEXITY_API_KEY=... uv run modal run cfpb/modal_eval.py --step frontier

# all tables in this README, written to results/cfpb/report.md
uv run modal run cfpb/modal_eval.py --step report
```

For runs longer than a few minutes, deploy the app and start the job with `.spawn()` instead of `modal run`. A `modal run` job stops when your laptop disconnects, even with `--detach` if the local client crashes. Training checkpoints go to the volume, so a restarted job resumes.

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

From the CFPB benchmark:

- **Deduplicate before you split.** 49% of the complaint narratives were copies of another one, mostly credit-repair template letters refiled with different redacted names. Without deduplication, the same letter appears in training and test.
- **Split by time, not at random.** A deployed model only sees complaints filed after its training data. A random split hides drift.
- **Small datasets memorize fast.** With 100 examples, validation loss was lowest at step 36 (0.44) and climbed to 0.70 by the end of training. Keeping the best checkpoint fixes it; the final checkpoint would have been the worse model.
- **"Reasoning off" isn't always off.** With a long prompt listing 132 labels, Claude still reasoned at effort `none` and sometimes spent its whole output budget, returning nothing. The CFPB runs use effort `high` with a 4,096-token budget. Haiku then averaged 260 reasoning tokens per complaint; Sonnet averaged 20.
- **A faster GPU isn't a free upgrade.** The same training code that runs cleanly on an L40S produced NaN gradients from step 60 on an H200 (requested as H100), so the adapter learned nothing. Check gradient norms in the first minutes of any run on new hardware.
- **Try the boring baseline.** TF-IDF + logistic regression trains in seconds and, at 20,000 examples, beats Haiku on product + issue.

## Caveats

- **Self-hosted cost** is GPU time during a run with 32 concurrent requests. It excludes idle time and cold starts (about 1–4 minutes). vLLM reported room for about 290 concurrent requests, so heavier traffic would lower the cost per request. Low traffic would raise it unless the server scales to zero.
- **Latency** includes the network round-trip to Modal's `us-east` region.
- **Banking77 is clean, short, single-intent data.** Real support tickets are longer, messier and sometimes cover several issues, so expect lower accuracy on them until the model is trained on that data.
- **Claude models ran with reasoning off on Banking77**, which is how you'd normally run a high-volume classifier. With reasoning on they may score higher, at a much higher cost. On CFPB they ran with reasoning effort `high`.
- **CFPB labels are chosen by the people filing**, and the test set is balanced across products. Accuracy on a company's own agent-labeled tickets, in their natural mix, will differ.
- **Claude was scored on 1,000 CFPB test complaints, not all 3,300**, to limit API cost. Every comparison table uses the same 1,000.

## Cost of this experiment

| Item | Approx. cost |
|---|---|
| LoRA training (L40S) | ~$0.15 of GPU time, plus a one-time image build and model download |
| Serving and evaluation (L4) | < $1 |
| Claude Haiku 5.5, full test set | ~$0.32 |
| Claude Sonnet 5.5, full test set | ~$6.62 |
| **Total** | **~$8–9** |

The CFPB benchmark cost about $4–5 of L40S time across the five training sizes (the 20,000-example run takes 82 minutes), under $1 of L4 evaluation, and $3.29 for Claude on 1,000 complaints ($0.46 Haiku, $2.82 Sonnet), excluding failed runs.

## Credits

- Dataset: Banking77, from Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), via the [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) mirror.
- Dataset: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/), published by the US Consumer Financial Protection Bureau. Narratives are published with consumer consent, with personal information removed.
- Base model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- Built with [Modal](https://modal.com), [vLLM](https://github.com/vllm-project/vllm), [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft).
