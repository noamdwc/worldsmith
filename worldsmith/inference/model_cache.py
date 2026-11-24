"""Load an R2 model snapshot, saving missing Hub downloads for later runs."""

import argparse
import json
import runpy
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import MODEL, MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    files = json.loads(args.plan.read_text())
    args.plan.unlink()
    # Reuse the same R2 transfer helpers as the other Colab caches.
    transfer = runpy.run_path(str(args.plan.parent / "scripts/setup-colab.py"))
    cache_dir = args.plan.parent / "hf-cache"
    directory = cache_dir / ("models--" + MODEL.replace("/", "--")) / "snapshots" / MODEL_REVISION
    directory.mkdir(parents=True, exist_ok=True)
    missing = [name for name, cache in files.items() if "put" in cache and not (directory / name).exists()]
    if missing:
        print("Downloading missing model files from Hugging Face", flush=True)
        snapshot_download(MODEL, revision=MODEL_REVISION, allow_patterns=missing, cache_dir=cache_dir)
    for name, cache in files.items():
        destination = directory / name
        if "get" in cache and not destination.exists():
            print(f"Restoring model file from R2: {name}", flush=True)
            temporary = destination.with_name(name + ".partial")
            transfer["download"](cache["get"], temporary)
            if transfer["cache_digest"](temporary).hexdigest() != cache["etag"]:
                raise RuntimeError(f"R2 checksum mismatch for {name}")
            temporary.replace(destination)
        elif "put" in cache:
            print(f"Saving model file to R2: {name}", flush=True)
            transfer["publish"](cache, destination)


if __name__ == "__main__":
    main()
