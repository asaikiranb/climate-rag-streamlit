"""Helpers to resolve locally cached Hugging Face model snapshots."""

from __future__ import annotations

import os
from pathlib import Path


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        roots.append(Path(hf_home))

    xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        roots.append(Path(xdg_cache) / "huggingface")

    roots.append(Path.home() / ".cache" / "huggingface")
    roots.append(Path(__file__).resolve().parent / ".runtime-cache" / "xdg" / "huggingface")
    return roots


def resolve_local_snapshot(model_name: str) -> str | None:
    if not model_name or "/" not in model_name:
        return None

    org, name = model_name.split("/", 1)
    repo_dir = f"models--{org}--{name}"

    for root in _cache_roots():
        model_root = root / "hub" / repo_dir
        ref_main = model_root / "refs" / "main"
        if not ref_main.exists():
            continue

        revision = ref_main.read_text(encoding="utf-8").strip()
        snapshot_path = model_root / "snapshots" / revision
        if not snapshot_path.exists():
            continue
        if (snapshot_path / "config.json").exists() or (snapshot_path / "modules.json").exists():
            return str(snapshot_path)
    return None
