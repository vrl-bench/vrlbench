
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SYNAPSE_SOURCE_COMMIT = "89dbbf8f617fdfb6d97b27324719db95da6ccdff"
SYNAPSE_SOURCE_URL = "https://github.com/ltzheng/Synapse.git"
SYNAPSE_RUNTIME_OVERLAY = "vrlbench/environments/synapse_overlay"
SYNAPSE_RUNTIME_FILE_SHA256 = {
    "synapse/envs/miniwob/action.py": (
        "2a282200300ef5813fc371a1eed2ab0bed936eb64895f9d1863e9f73f02b0db4"
    ),
    "synapse/envs/miniwob/environment.py": (
        "4c05c2eeeea8632bdaa05743c8cfecb4fae2da52d72cd5b5fad149bd64775b49"
    ),
    "synapse/envs/miniwob/instance.py": (
        "14696ab0a111aa2a58f4f6232a519e7e0f7115aade96b873672cbe526b4333e8"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_overlay_root() -> Path:
    return Path(__file__).resolve().parent / "synapse_overlay"


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ImportError("Synapse checkout has no readable Git HEAD")
    return completed.stdout.strip()


def synapse_runtime_issues(root: Path) -> list[str]:
    """Return content mismatches for the required MiniWoB runtime overlay."""
    issues: list[str] = []
    for relative, expected in SYNAPSE_RUNTIME_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            issues.append(f"missing {relative}")
            continue
        actual = _sha256(path)
        if actual != expected:
            issues.append(
                f"{relative} sha256 mismatch: expected {expected}, got {actual}"
            )
    return issues


def synapse_dependency_contract() -> dict[str, Any]:
    """Return and validate the released dependency reproduction contract."""
    overlay_root = _runtime_overlay_root()
    overlay_issues = synapse_runtime_issues(overlay_root)
    if overlay_issues:
        raise ImportError(
            "released Synapse MiniWoB runtime overlay is missing or changed: "
            + "; ".join(overlay_issues)
        )
    return {
        "source": "Synapse",
        "source_url": SYNAPSE_SOURCE_URL,
        "base_commit": SYNAPSE_SOURCE_COMMIT,
        "working_tree_contract": "base_commit_plus_final_file_overlay",
        "runtime_overlay": SYNAPSE_RUNTIME_OVERLAY,
        "runtime_overlay_file_sha256": dict(SYNAPSE_RUNTIME_FILE_SHA256),
        "role": "benchmark_actor_and_preprocessing",
    }


def configure_synapse_import_path() -> Path:
    """Validate and add the overlaid Synapse checkout to the import path."""
    configured = os.environ.get("SYNAPSE_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        source = "SYNAPSE_ROOT"
    else:
        raise ImportError(
            "Set SYNAPSE_ROOT to the public Synapse dependency checkout before "
            "running MiniWoB. The dependency is not installed inside vrlbench."
        )

    package = root / "synapse"
    if not (package / "envs" / "miniwob").is_dir() or not (
        package / "envs" / "mind2web" / "env_utils.py"
    ).is_file():
        raise ImportError(
            "Synapse benchmark utilities were not found under "
            f"{root} (resolved from {source}). Configure SYNAPSE_ROOT "
            "to point to the public dependency checkout."
        )

    synapse_dependency_contract()
    actual_head = _git_head(root)
    if actual_head != SYNAPSE_SOURCE_COMMIT:
        raise ImportError(
            "Synapse checkout HEAD differs from the required base commit: "
            f"expected {SYNAPSE_SOURCE_COMMIT}, got {actual_head}"
        )
    issues = synapse_runtime_issues(root)
    if issues:
        raise ImportError(
            "Synapse MiniWoB runtime overlay contract failed: "
            + "; ".join(issues)
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
