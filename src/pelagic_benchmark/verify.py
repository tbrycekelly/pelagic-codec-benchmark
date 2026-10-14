from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from .storage import SCHEMAS, sha256


def verify(run_directory: str | Path) -> list[str]:
    directory = Path(run_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Run manifest is not marked completed.")
    if sha256(directory / "config.toml") != manifest.get("config_file_sha256"):
        raise ValueError("config.toml hash does not match manifest.")
    checked: list[str] = []
    for name, details in manifest.get("artifacts", {}).items():
        if name not in SCHEMAS:
            raise ValueError(f"Manifest contains unknown table {name!r}.")
        path = directory / f"{name}.parquet"
        if not path.is_file():
            raise ValueError(f"Missing artifact: {path.name}")
        if sha256(path) != details.get("sha256"):
            raise ValueError(f"Hash mismatch for {path.name}")
        table = pq.read_table(path)
        if not table.schema.equals(SCHEMAS[name], check_metadata=False):
            raise ValueError(f"Schema mismatch for {path.name}")
        if table.num_rows != details.get("rows"):
            raise ValueError(f"Row-count mismatch for {path.name}")
        checked.append(path.name)
    if set(manifest.get("artifacts", {})) != set(SCHEMAS):
        raise ValueError("Manifest does not describe the complete required table set.")
    return checked
