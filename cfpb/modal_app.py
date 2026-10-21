"""CFPB complaint data on Modal: download → prepare → baseline. Data lives in the `tinytriage-data` volume."""

import json
from pathlib import Path

import modal

EXPORTS = {
    "jul_aug2025": "CCDB_Export_12_July_2025_through_August_2025",
    "sep_oct2025": "CCDB_Export_13_September_2025_through_October_2025",
    "nov_dec2025": "CCDB_Export_14_November_2025_through_December_2025",
}
BASE_URL = "https://files.consumerfinance.gov/f/documents"
HERE = Path(__file__).resolve().parent
LOCAL_RESULTS = HERE.parent / "results" / "cfpb"

app = modal.App("tinytriage-cfpb")
data = modal.Volume.from_name("tinytriage-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("scikit-learn")
    .env({"CFPB_DATA": "/data", "PYTHONPATH": "/root/cfpb"})
    .add_local_dir(HERE, "/root/cfpb", ignore=["__pycache__", "modal_app.py"])
)


@app.function(image=image, volumes={"/data": data}, cpu=2, memory=4096, timeout=60 * 60)
def download() -> list[str]:
    import urllib.request
    import zipfile

    raw = Path("/data/raw/cfpb")
    fetched = []
    for folder, name in EXPORTS.items():
        target = raw / folder
        if any(target.glob("*.csv")):
            continue
        target.mkdir(parents=True, exist_ok=True)
        archive = raw / f"{folder}.zip"
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.zip", archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        archive.unlink()
        fetched.append(folder)
    data.commit()
    return fetched


@app.function(image=image, volumes={"/data": data}, cpu=4, memory=8192, timeout=60 * 60)
def prepare() -> dict:
    import prepare as step

    stats = step.main()
    data.commit()
    return stats


@app.function(image=image, volumes={"/data": data}, cpu=8, memory=16384, timeout=60 * 60)
def baseline() -> list[dict]:
    import baseline as step

    results = step.main()
    data.commit()
    return results


@app.local_entrypoint()
def main(step: str = "all"):
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    if step in ("all", "download"):
        print("downloaded:", download.remote() or "already present")
    if step in ("all", "prepare"):
        stats = prepare.remote()
        (LOCAL_RESULTS / "stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2))
    if step in ("all", "baseline"):
        results = baseline.remote()
        (LOCAL_RESULTS / "baseline_tfidf.json").write_text(json.dumps(results, indent=2))
        for r in results:
            print(f"n={r['train_size']:6d}  product {r['product_accuracy']:6.1%}  "
                  f"product+issue {r['pair_accuracy']:6.1%}  | on 1,000 sample: product "
                  f"{r['sample1000_product_accuracy']:6.1%}  product+issue {r['sample1000_pair_accuracy']:6.1%}")
