from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import psutil


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_state(path: Path) -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def provenance(project_root: Path) -> dict[str, object]:
    pelagia_version = package_version("pelagia")
    pelagia_location: Path | None = None
    if pelagia_version:
        try:
            import Pelagia
            pelagia_location = Path(Pelagia.__file__).resolve().parent.parent
        except Exception:
            pass
    return {
        "project": {"version": package_version("pelagic-codec-benchmark"), **git_state(project_root)},
        "pelagia": {"version": pelagia_version, **(git_state(pelagia_location) if pelagia_location else {"commit": None, "dirty": None})},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "kernel": platform.release(),
        "hostname": platform.node(),
        "cpu_model": platform.processor() or os.uname().machine,
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "total_ram_bytes": psutil.virtual_memory().total,
        "libraries": {name: package_version(name) for name in ("numpy", "Pillow", "opencv-python-headless", "pyarrow", "pelagia")},
    }
