"""
One-shot script to download the MLSys26 contest dataset into a Modal volume.
Run with: modal run download_dataset.py

To clean up a failed partial download first, run:
    modal volume rm flashinfer-trace /mlsys26-contest
"""

import modal

app = modal.App("download-dataset")
vol = modal.Volume.from_name("flashinfer-trace")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster downloads
)

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=3600,
)
def download_dataset():
    import shutil
    import os
    from huggingface_hub import snapshot_download

    dest = "/data"

    # Clean up any partial download from a previous failed run
    if os.path.exists(dest):
        print(f"Removing existing directory {dest} before re-downloading...")
        shutil.rmtree(dest)

    print("Downloading dataset from HuggingFace...")
    snapshot_download(
        repo_id="flashinfer-ai/mlsys26-contest",
        repo_type="dataset",
        local_dir=dest,
    )

    vol.commit()
    print("Done! Dataset is now in the flashinfer-trace volume at /data")


@app.local_entrypoint()
def main():
    download_dataset.remote()
