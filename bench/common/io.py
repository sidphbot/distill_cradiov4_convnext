"""I/O helpers: JSON/JSONL, run metadata, git hash."""

import json
import subprocess
import time
from pathlib import Path


def save_json(data: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(data: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_git_hash() -> str:
    """Return short git revision hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_run_meta(out_dir: str, args) -> dict:
    """Write run_meta.json with timestamp, git hash, and CLI args."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": get_git_hash(),
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
    }
    save_json(meta, str(Path(out_dir) / "run_meta.json"))
    return meta
