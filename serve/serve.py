import os
import subprocess

import modal

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = "qwen2.5-1.5b-banking77"
PORT = 8000

app = modal.App("banking77-serve")

# FlashInfer's sampler JIT-compiles kernels at startup, which needs nvcc that the slim image lacks.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("banking77-adapters")
api_key = modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "")})


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": hf_cache, "/adapters": adapters},
    secrets=[api_key],
    scaledown_window=5 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve():
    # One server answers as both models: "base" is plain Qwen, "banking77" applies the LoRA adapter.
    subprocess.Popen([
        "vllm", "serve", BASE_MODEL,
        "--served-model-name", "base",
        "--enable-lora",
        "--lora-modules", f"banking77=/adapters/{ADAPTER_NAME}",
        "--max-lora-rank", "16",
        "--max-model-len", "2048",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
    ])
