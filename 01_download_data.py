"""Download required AIDev tables."""
import os
from huggingface_hub import hf_hub_download

REPO = "hao-li/AIDev"
DEST = "data"
os.makedirs(DEST, exist_ok=True)

FILES = ["pull_request.parquet", "pr_commits.parquet"]

for f in FILES:
    print(f"Downloading {f}...")
    hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset", local_dir=DEST)

print("Done.")
