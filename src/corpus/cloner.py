"""Shallow-clone repos from config/repos.yaml for corpus building.

Stores cloned repos under STRIFF_DATA_DIR (default: ./data)
so they persist across training runs.
"""

import os
import yaml
from git import Repo
from pathlib import Path
from tqdm import tqdm

# Default data directory — all training data lives here
STRIFF_DATA_DIR = os.environ.get("STRIFF_DATA_DIR", "./data")

# GitHub token for authenticated cloning (avoids rate limits)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def load_repos(config_path: str = "config/repos.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def clone_repo(url: str, ref: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Inject token into URL for authenticated cloning
    clone_url = url
    if GITHUB_TOKEN and "github.com" in url:
        clone_url = url.replace("https://", f"https://{GITHUB_TOKEN}@")
    Repo.clone_from(clone_url, str(dest), branch=ref, depth=1)
    return dest


def clone_all(output_dir: str | None = None) -> dict[str, Path]:
    if output_dir is None:
        output_dir = os.path.join(STRIFF_DATA_DIR, "corpus/repos")
    repos = load_repos()
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    results = {}
    all_entries = []
    for lang, entries in repos.items():
        for entry in entries:
            name = entry["url"].rstrip("/").split("/")[-1]
            all_entries.append((lang, name, entry))

    for lang, name, entry in tqdm(all_entries, desc="Cloning repos"):
        dest = base / lang / name
        try:
            clone_repo(entry["url"], entry["ref"], dest)
            results[f"{lang}/{name}"] = dest
        except Exception as e:
            print(f"Failed to clone {entry['url']}: {e}")
    return results


if __name__ == "__main__":
    paths = clone_all()
    print(f"Cloned {len(paths)} repos")
    for key, path in sorted(paths.items()):
        print(f"  {key}: {path}")
