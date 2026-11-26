"""Resolve and authenticate the catalog, indexes and encoder as a single bundle."""
from hashlib import file_digest
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
ASSETS = json.loads((ROOT / "assets_v3.json").read_text(encoding="utf-8"))


class AssetError(RuntimeError):
    """An incomplete, mismatched or inaccessible deployment bundle."""


def verify_assets(directory, files=ASSETS):
    directory = Path(directory).expanduser().resolve()
    for name in files:
        expected = ASSETS[name]
        path = directory / name
        try:
            if path.stat().st_size == expected["bytes"]:
                with path.open("rb") as stream:
                    if file_digest(stream, "sha256").hexdigest() == expected["sha256"]:
                        continue
        except OSError:
            pass
        raise AssetError(
            f"Missing or mismatched asset: {name} in {directory}. "
            "Point RECIPE_SYSTEM_DIR to a complete V3 bundle or verify the Hub commit."
        )
    return directory


def resolve_assets(local=None, repo=None, revision=None, token=None, *, files=ASSETS):
    if local:
        return verify_assets(local, files)
    if repo or revision:
        if not repo or not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
            raise AssetError(
                "Set SCARLET_HF_REPO to the model repository ID and "
                "SCARLET_HF_REVISION to its full 40-character commit SHA."
            )
        try:
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                repo_id=repo, revision=revision.lower(), repo_type="model",
                token=token or False, allow_patterns=list(files),
            )
        except Exception as error:
            raise AssetError(
                f"Could not fetch the pinned assets ({type(error).__name__}). "
                "Check network access, the repository and commit, and HF_TOKEN for a private repo."
            ) from error
        return verify_assets(snapshot, files)
    local_candidates = (ROOT / "catalog.jsonl", ROOT.parent / "local-assets/catalog.jsonl",
                        ROOT.parent.parent / "recipe_system_v3/catalog.jsonl")
    for catalog in local_candidates:
        if catalog.is_file():
            return verify_assets(catalog.parent, files)
    raise AssetError(
        "No local V3 catalog found. Put the assets in the release folder's local-assets directory, "
        "set RECIPE_SYSTEM_DIR to your extracted bundle, or configure the pinned Hugging Face repository."
    )


def configured_assets(*, files=ASSETS):
    return resolve_assets(
        os.getenv("RECIPE_SYSTEM_DIR"), os.getenv("SCARLET_HF_REPO"),
        os.getenv("SCARLET_HF_REVISION"), os.getenv("HF_TOKEN"), files=files,
    )

