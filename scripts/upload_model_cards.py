#!/usr/bin/env python3
"""Upload model cards (README.md) to the HuggingFace Hub repos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS_DIR = ROOT / "python" / "model_cards"

REPOS = {
    "pusht": ("abdelstark/lewm-rs-pusht", CARDS_DIR / "README_pusht.md"),
    "so100": ("abdelstark/lewm-rs-so100", CARDS_DIR / "README_so100.md"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=[*REPOS.keys(), "all"],
        help="Which model card to upload: pusht, so100, or all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without uploading")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN environment variable required", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub required: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    targets = list(REPOS.keys()) if args.target == "all" else [args.target]
    failures = 0

    for target in targets:
        repo_id, card_path = REPOS[target]
        if not card_path.exists():
            print(f"[{target}] Card not found: {card_path}", file=sys.stderr)
            failures += 1
            continue

        print(f"[{target}] Uploading {card_path.name} → {repo_id}/README.md")
        if args.dry_run:
            print(f"[{target}] DRY RUN — skipping upload")
            continue

        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add model card for {target}",
            )
            print(f"[{target}] Uploaded successfully")
        except Exception as e:
            print(f"[{target}] Upload failed: {e}", file=sys.stderr)
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
