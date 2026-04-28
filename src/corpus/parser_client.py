"""Calls clarpse Docker API to parse repos into structural models.

The clarpse server exposes POST /parse with two modes:
- JSON: Content-Type application/json with {language, files: [{path, content}]}
- ZIP:  Content-Type application/zip with ?lang= query param

This client uses the JSON mode to send source files.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional


CLARPSE_API_URL = os.environ.get("CLARPSE_API_URL", "http://localhost:8080")

# Extensions per language for file discovery
EXTENSIONS = {
    "java": {".java"},
    "python": {".py"},
    "typescript": {".ts", ".tsx"},
}

# Map to clarpse's Lang names
LANG_MAP = {
    "java": "java",
    "python": "python",
    "typescript": "typescript",
}


def health_check() -> bool:
    """Check if clarpse server is healthy."""
    try:
        r = requests.get(f"{CLARPSE_API_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def parse_with_clarpse(repo_path: str, language: str) -> Optional[dict]:
    """Parse a repo directory using the clarpse Docker API (JSON mode).

    Reads all source files for the given language, sends them as a JSON
    payload to POST /parse, and returns the raw clarpse response dict.

    Returns None on failure.
    """
    exts = EXTENSIONS.get(language, set())
    lang_param = LANG_MAP.get(language, language)
    files = []

    # For TypeScript, include tsconfig.json (required by clarpse's TS parser)
    tsconfig_paths = ["tsconfig.json"] if language == "typescript" else []
    for tc in tsconfig_paths:
        tc_path = os.path.join(repo_path, tc)
        if os.path.exists(tc_path):
            try:
                with open(tc_path, 'r', encoding='utf-8', errors='ignore') as f:
                    files.append({"path": tc, "content": f.read()})
            except Exception:
                pass

    for root, dirs, filenames in os.walk(repo_path):
        # Skip hidden and dependency directories
        dirs[:] = [d for d in dirs if not d.startswith('.')
                   and d not in ("node_modules", "__pycache__", "build",
                                 "target", "dist", ".git", "venv", ".venv")]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in exts:
                continue
            # Skip .d.ts and test files to reduce noise
            if language == "typescript" and (fname.endswith(".d.ts") or
                                            fname.endswith(".spec.ts") or
                                            fname.endswith(".test.ts")):
                continue
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_path)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                files.append({"path": rel_path, "content": content})
            except Exception:
                continue

    if not files:
        print(f"  No {language} source files found in {repo_path}")
        return None

    payload = {
        "language": lang_param,
        "files": files,
    }

    try:
        response = requests.post(
            f"{CLARPSE_API_URL}/parse",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  Clarpse parse failed for {repo_path} ({language}): {e}")
        return None


def parse_repo(repo_path: Path, language: str) -> Optional[dict]:
    """Parse a single repo directory, invoking clarpse API."""
    return parse_with_clarpse(str(repo_path), language)
