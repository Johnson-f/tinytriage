import modal

app = modal.App("hello-gpu")


@app.function(gpu="T4", timeout=120)
def check_gpu():
    import subprocess

    subprocess.run(["nvidia-smi"], check=True)


@app.local_entrypoint()
def main():
    check_gpu.remote()
