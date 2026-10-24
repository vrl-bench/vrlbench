
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, distributions, version
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


WEBSHOP_SOURCE_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
WEBSHOP_SOURCE_REPOSITORY = "https://github.com/princeton-nlp/webshop"
WEBSHOP_DATA_MIRROR_REPOSITORY = (
    "https://huggingface.co/datasets/YWZBrandon/webshop-data"
)
WEBSHOP_DATA_MIRROR_REVISION = "ce990fff5aee388db2706f07820c578ab68e0453"
WEBSHOP_OFFICIAL_DATA_FILE_IDS = {
    "data/items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
    "data/items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    "data/items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}
WEBSHOP_DATA_SOURCE = {
    "official_setup": {
        "source_commit": WEBSHOP_SOURCE_COMMIT,
        "google_drive_file_ids": WEBSHOP_OFFICIAL_DATA_FILE_IDS,
    },
    "content_addressed_mirror": {
        "repository": WEBSHOP_DATA_MIRROR_REPOSITORY,
        "revision": WEBSHOP_DATA_MIRROR_REVISION,
    },
}
WEBSHOP_RUNTIME_FILE_SHA256 = {
    "requirements.txt": "b4e83b7e9c670c724215fcb79e43f52f591de354a8f70870ba4553c88de26d2d",
    "baseline_models/env.py": "88e00b6dae4bb8cf1c042059104ef83f0cdf8704d61a5ac41d1810f10d1f7eac",
    "web_agent_site/app.py": "f9393af973fd2d0343c8b68e89572e6139ebc8d6a2309a145c554a80284e81cb",
    "web_agent_site/utils.py": "bdd6df3b869cf3fe028169add3611d1b811ae663ae4f79520617c26b20614970",
    "web_agent_site/engine/engine.py": "9b86ffc82951124f5c4a7eb8103de14100dbee11ba8869951ed6fb8feaeef804",
    "web_agent_site/engine/goal.py": "9703c8583244e2041182eb14856186b998145ab16ce2ba9aef8cb680877c8ed5",
    "web_agent_site/engine/normalize.py": "1962113ac9e209a636d4f181b2c70544bbea59b03adca9847e48287158d794fb",
    "web_agent_site/envs/web_agent_text_env.py": "f4efee238c2a69ad76ff5372716a16051c67ea0351c7006491834822ea9afda3",
    "search_engine/run_indexing.sh": "96f0a2323bf6fda0eb0549fd66265c0cb05f988813f8181110761dc43b88ea0c",
}
WEBSHOP_FULL_DATA_FILES = (
    "data/items_shuffle.json",
    "data/items_ins_v2.json",
    "data/items_human_ins.json",
)
WEBSHOP_FULL_DATA_SHA256 = {
    "data/items_shuffle.json": "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    "data/items_ins_v2.json": "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e",
    "data/items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}
WEBSHOP_FULL_INDEX_MARKER = "search_engine/indexes/vex2_full_index_meta.json"
WEBSHOP_RUNTIME_PROFILE = "pinned_compatible_py310_java11"
WEBSHOP_INDEX_CONTENT_SCHEMA = "webshop_lucene_latest_commit_sha256_v1"
WEBSHOP_BUILD_ENVIRONMENT_SCHEMA = "webshop_build_environment_v1"
WEBSHOP_SOURCE_DATA_INDEX_SCHEMA = "webshop_source_data_index_sha256_v1"
WEBSHOP_CANONICAL_DIGEST = "sha256_canonical_json_v1"
WEBSHOP_FULL_INDEX_PROFILE = {
    "schema_version": 2,
    "source_repository": WEBSHOP_SOURCE_REPOSITORY,
    "source_commit": WEBSHOP_SOURCE_COMMIT,
    "source_runtime_file_sha256": WEBSHOP_RUNTIME_FILE_SHA256,
    "data_profile": "full",
    "data_source": WEBSHOP_DATA_SOURCE,
    "data_file_sha256": WEBSHOP_FULL_DATA_SHA256,
    "index_directory": "search_engine/indexes",
    "runtime_profile": WEBSHOP_RUNTIME_PROFILE,
    "raw_catalog_products": 1_181_436,
    "converted_documents": 1_181_430,
    "empty_documents": 60,
    "indexed_documents": 1_181_370,
}
WEBSHOP_FULL_INDEX_INTEGER_FIELDS = frozenset(
    {
        "schema_version",
        "raw_catalog_products",
        "converted_documents",
        "empty_documents",
        "indexed_documents",
    }
)
WEBSHOP_FULL_INDEX_NUM_CONVERTED_DOCUMENTS = WEBSHOP_FULL_INDEX_PROFILE[
    "converted_documents"
]
WEBSHOP_FULL_INDEX_NUM_DOCUMENTS = WEBSHOP_FULL_INDEX_PROFILE[
    "indexed_documents"
]
WEBSHOP_OFFICIAL_TEST_GOAL_START = 0
WEBSHOP_OFFICIAL_TEST_GOAL_END = 499
WEBSHOP_RUNTIME_PYTHON = [3, 10]
WEBSHOP_RUNTIME_JAVA_MAJOR = 11
WEBSHOP_REQUIREMENTS_LOCK_RELATIVE = (
    "requirements/webshop-official-py310.lock.txt"
)
WEBSHOP_REQUIREMENTS_LOCK = (
    Path(__file__).resolve().parents[1] / WEBSHOP_REQUIREMENTS_LOCK_RELATIVE
)
WEBSHOP_RUNTIME_REQUIRED_DISTRIBUTIONS = frozenset(
    {
        "beautifulsoup4",
        "cleantext",
        "en-core-web-sm",
        "flask",
        "gym",
        "numpy",
        "pyserini",
        "rank-bm25",
        "rich",
        "spacy",
        "thefuzz",
        "torch",
        "tqdm",
        "transformers",
        "werkzeug",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _locked_versions(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse exact direct dependencies from the pip-compatible lock file."""

    if not path.is_file():
        return {}, [f"missing requirements lock {path}"]

    versions: dict[str, str] = {}
    issues: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--"):
            if line != "--extra-index-url https://download.pytorch.org/whl/cpu":
                issues.append(
                    f"unsupported requirements option on line {line_number}: {line}"
                )
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            issues.append(f"invalid locked requirement on line {line_number}: {exc}")
            continue

        package = canonicalize_name(requirement.name)
        if package in versions:
            issues.append(f"duplicate locked distribution {package}")
            continue
        if requirement.marker is not None:
            issues.append(f"environment marker is not allowed for {package}")
            continue

        if requirement.url:
            filename = Path(unquote(urlparse(requirement.url).path)).name
            try:
                wheel_name, wheel_version, _build, _tags = parse_wheel_filename(
                    filename
                )
            except ValueError as exc:
                issues.append(f"direct URL for {package} is not a pinned wheel: {exc}")
                continue
            if canonicalize_name(wheel_name) != package:
                issues.append(
                    f"direct URL distribution mismatch for {package}: {wheel_name}"
                )
                continue
            if "#sha256=" not in requirement.url:
                issues.append(f"direct URL for {package} lacks a sha256 fragment")
                continue
            locked_version = str(wheel_version)
        else:
            specifiers = list(requirement.specifier)
            if (
                len(specifiers) != 1
                or specifiers[0].operator != "=="
                or "*" in specifiers[0].version
            ):
                issues.append(f"distribution {package} is not exactly pinned")
                continue
            locked_version = specifiers[0].version

        versions[package] = locked_version

    missing = sorted(WEBSHOP_RUNTIME_REQUIRED_DISTRIBUTIONS - versions.keys())
    unexpected = sorted(versions.keys() - WEBSHOP_RUNTIME_REQUIRED_DISTRIBUTIONS)
    if missing:
        issues.append("requirements lock is missing: " + ", ".join(missing))
    if unexpected:
        issues.append("requirements lock has unexpected entries: " + ", ".join(unexpected))
    return versions, issues


WEBSHOP_RUNTIME_PACKAGE_VERSIONS, WEBSHOP_REQUIREMENTS_LOCK_ISSUES = (
    _locked_versions(WEBSHOP_REQUIREMENTS_LOCK)
)
WEBSHOP_REQUIREMENTS_LOCK_SHA256 = (
    _sha256(WEBSHOP_REQUIREMENTS_LOCK)
    if WEBSHOP_REQUIREMENTS_LOCK.is_file()
    else ""
)


def _git_output(path: Path, *args: str) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def _normalized_repository_url(value: str) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("git@github.com:"):
        text = "https://github.com/" + text[len("git@github.com:") :]
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/")


def _webshop_source_data_issues(path: Path) -> list[str]:
    """Return source and full-data issues that are checkable before indexing."""

    issues: list[str] = []
    if not (path / ".git").exists():
        issues.append("checkout lacks .git provenance")
    else:
        returncode, top_level = _git_output(path, "rev-parse", "--show-toplevel")
        if returncode != 0 or not top_level:
            issues.append("source checkout is not a readable Git worktree")
        elif Path(top_level).expanduser().resolve() != path:
            issues.append(
                "source checkout root mismatch: "
                f"expected {path}, got {Path(top_level).expanduser().resolve()}"
            )
        returncode, actual = _git_output(path, "rev-parse", "HEAD")
        if returncode != 0 or actual != WEBSHOP_SOURCE_COMMIT:
            issues.append(
                "source commit mismatch: "
                f"expected {WEBSHOP_SOURCE_COMMIT}, got {actual or 'unknown'}"
            )
        returncode, origin = _git_output(path, "remote", "get-url", "origin")
        if (
            returncode != 0
            or _normalized_repository_url(origin)
            != _normalized_repository_url(WEBSHOP_SOURCE_REPOSITORY)
        ):
            issues.append(
                "source repository mismatch: "
                f"expected {WEBSHOP_SOURCE_REPOSITORY}, got {origin or 'unknown'}"
            )
        returncode, worktree_status = _git_output(
            path,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        if returncode != 0 or worktree_status:
            issues.append("source worktree is not clean")
    for relative, expected in WEBSHOP_RUNTIME_FILE_SHA256.items():
        candidate = path / relative
        if not candidate.is_file():
            issues.append(f"missing runtime file {relative}")
        elif _sha256(candidate) != expected:
            issues.append(f"runtime file hash mismatch {relative}")
    for relative in WEBSHOP_FULL_DATA_FILES:
        candidate = path / relative
        if not candidate.is_file():
            issues.append(f"missing full-data file {relative}")
        elif _sha256(candidate) != WEBSHOP_FULL_DATA_SHA256[relative]:
            issues.append(f"full-data file hash mismatch {relative}")
    return issues


def webshop_lucene_committed_file_names(root: str | Path) -> tuple[str, ...]:
    """Return the files referenced by Lucene's latest committed index state."""

                                                                                
                                                                        
    from pyserini.pyclass import autoclass

    index_root = Path(root).expanduser().resolve() / "search_engine" / "indexes"
    fs_directory = autoclass("org.apache.lucene.store.FSDirectory")
    paths = autoclass("java.nio.file.Paths")
    directory_reader = autoclass("org.apache.lucene.index.DirectoryReader")
    directory = fs_directory.open(paths.get(str(index_root)))
    try:
        commits = directory_reader.listCommits(directory)
        if commits.size() < 1:
            raise RuntimeError("Lucene index has no committed state")
        latest_commit = commits.get(commits.size() - 1)
        names = tuple(str(name) for name in latest_commit.getFileNames().toArray())
    finally:
        directory.close()
    if len(names) != len(set(names)):
        raise RuntimeError("Lucene latest commit reports duplicate file names")
    return tuple(sorted(names))


def _stable_file_manifest_entry(index_root: Path, name: str) -> dict[str, Any]:
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
        raise RuntimeError(f"unsafe Lucene committed file name {name!r}")
    candidate = index_root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"missing regular Lucene committed file {name}")

    before = candidate.stat()
    digest = _sha256(candidate)
    after = candidate.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"Lucene committed file changed while hashing: {name}")
    return {
        "path": name,
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def webshop_index_content_provenance(root: str | Path) -> dict[str, Any]:
    """Hash every file referenced by the latest Lucene commit."""

    path = Path(root).expanduser().resolve()
    index_root = path / "search_engine" / "indexes"
    names = tuple(sorted(webshop_lucene_committed_file_names(path)))
    if len(names) != len(set(names)):
        raise RuntimeError("Lucene latest commit reports duplicate file names")
    if not names or not any(name.startswith("segments_") for name in names):
        raise RuntimeError("Lucene latest commit lacks a segments file")
    marker_name = Path(WEBSHOP_FULL_INDEX_MARKER).name
    if marker_name in names or "write.lock" in names:
        raise RuntimeError("Lucene committed-file set contains non-index metadata")

    files = [_stable_file_manifest_entry(index_root, name) for name in names]
    identity: dict[str, Any] = {
        "schema_version": WEBSHOP_INDEX_CONTENT_SCHEMA,
        "file_selection": "lucene_latest_index_commit",
        "file_hash_algorithm": "sha256",
        "root_digest_algorithm": WEBSHOP_CANONICAL_DIGEST,
        "file_count": len(files),
        "total_bytes": sum(int(record["size_bytes"]) for record in files),
        "files": files,
    }
    return {
        **identity,
        "root_sha256": _canonical_json_sha256(identity),
    }


def _installed_distribution_manifest() -> list[dict[str, str]]:
    # Use the same search path as imports and metadata.version(). A venv with
    # --system-site-packages can legitimately inherit native dependencies.
    resolved: dict[str, dict[str, str]] = {}
    for distribution in distributions():
        raw_name = str(distribution.metadata.get("Name") or "").strip()
        raw_version = str(distribution.version or "").strip()
        if not raw_name or not raw_version:
            raise RuntimeError("installed distribution metadata is incomplete")
        name = canonicalize_name(raw_name)
        resolved.setdefault(name, {"name": name, "version": raw_version})
    if not resolved:
        raise RuntimeError("installed distribution manifest is empty")
    return sorted(resolved.values(), key=lambda record: record["name"])


def _java_runtime_identity(
    java_executable: str | Path | None = None,
) -> dict[str, Any]:
    java = str(java_executable or shutil.which("java") or "").strip()
    if not java:
        raise RuntimeError("Java runtime for Lucene is not found")
    try:
        completed = subprocess.run(
            [java, "-version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RuntimeError(f"Java runtime for Lucene cannot be executed: {exc}") from exc
    version_lines = (completed.stderr or completed.stdout).splitlines()
    first_line = " ".join(version_lines[0].split()) if version_lines else ""
    match = re.search(r'version "(?P<major>\d+)', first_line)
    if completed.returncode != 0 or not match:
        raise RuntimeError(f"Java runtime version cannot be determined: {java}")
    return {
        "version": first_line,
        "major": int(match.group("major")),
    }


def webshop_build_environment_provenance(
    java_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Return a path-free hash of the complete resolved build environment."""

    installed = _installed_distribution_manifest()
    identity: dict[str, Any] = {
        "schema_version": WEBSHOP_BUILD_ENVIRONMENT_SCHEMA,
        "runtime_profile": WEBSHOP_RUNTIME_PROFILE,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "user_site_enabled": not bool(sys.flags.no_user_site),
        },
        "java": _java_runtime_identity(java_executable),
        "requirements": {
            "kind": "pinned_compatible_runtime_constraints",
            "relative_path": WEBSHOP_REQUIREMENTS_LOCK_RELATIVE,
            "sha256": WEBSHOP_REQUIREMENTS_LOCK_SHA256,
        },
        "installed_distribution_count": len(installed),
        "installed_distributions": installed,
    }
    return {
        **identity,
        "identity_sha256": _canonical_json_sha256(identity),
    }


def _source_data_index_identity(index_root_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": WEBSHOP_SOURCE_DATA_INDEX_SCHEMA,
        "source": {
            "repository": WEBSHOP_SOURCE_REPOSITORY,
            "commit": WEBSHOP_SOURCE_COMMIT,
            "runtime_file_sha256": dict(WEBSHOP_RUNTIME_FILE_SHA256),
        },
        "data": {
            "profile": "full",
            "source": WEBSHOP_DATA_SOURCE,
            "file_sha256": dict(WEBSHOP_FULL_DATA_SHA256),
        },
        "index": {
            "profile": "full",
            "committed_files_root_sha256": index_root_sha256,
        },
    }


def webshop_full_index_marker_payload(root: str | Path) -> dict[str, Any]:
    """Build the marker payload from the completed index and live environment."""

    index_content = webshop_index_content_provenance(root)
    build_environment = webshop_build_environment_provenance()
    source_data_index = _source_data_index_identity(index_content["root_sha256"])
    return {
        **json.loads(json.dumps(WEBSHOP_FULL_INDEX_PROFILE)),
        "index_content": index_content,
        "build_environment": build_environment,
        "source_data_index_identity": source_data_index,
        "source_data_index_root_sha256": _canonical_json_sha256(source_data_index),
    }


def write_webshop_full_index_marker(root: str | Path) -> Path:
    """Atomically write provenance after a successful full-index build."""

    path = Path(root).expanduser().resolve()
    marker = path / WEBSHOP_FULL_INDEX_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = webshop_full_index_marker_payload(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker.parent.parent,
            prefix=f".{marker.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, marker)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return marker


def _read_webshop_full_index_marker(
    marker: Path,
) -> tuple[object, str]:
    """Read one stable marker snapshot and hash the exact parsed bytes."""

    if marker.is_symlink() or not marker.is_file():
        raise FileNotFoundError(marker)
    before = marker.stat()
    raw_payload = marker.read_bytes()
    after = marker.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError("full-search-index marker changed while reading")
    payload = json.loads(raw_payload.decode("utf-8"))
    marker_sha256 = hashlib.sha256(raw_payload).hexdigest()
    return payload, marker_sha256


def _marker_digest_is_valid(
    payload: Any,
    *,
    digest_key: str,
) -> bool:
    if not isinstance(payload, dict):
        return False
    digest = payload.get(digest_key)
    identity = {key: value for key, value in payload.items() if key != digest_key}
    return isinstance(digest, str) and digest == _canonical_json_sha256(identity)


def _build_environment_is_runtime_compatible(
    build_environment: dict[str, Any],
    runtime_environment: dict[str, Any],
) -> bool:
    """Require compatible native runtimes, not an identical build machine.

    The index's bytes are verified separately. Java/Python patch releases,
    packaging tools, and comments in a requirements file do not change those
    bytes and are not reasons to reject an otherwise valid installation.
    """

    for field in (
        "schema_version",
        "runtime_profile",
    ):
        if build_environment.get(field) != runtime_environment.get(field):
            return False
    for environment in (build_environment, runtime_environment):
        python = environment.get("python")
        java = environment.get("java")
        if (
            not isinstance(python, dict)
            or python.get("implementation") != "CPython"
            or not isinstance(python.get("version_info"), list)
            or python["version_info"][:2] != WEBSHOP_RUNTIME_PYTHON
            or not isinstance(java, dict)
            or type(java.get("major")) is not int
            or java["major"] != WEBSHOP_RUNTIME_JAVA_MAJOR
        ):
            return False

    build_records = build_environment.get("installed_distributions")
    runtime_records = runtime_environment.get("installed_distributions")
    if not isinstance(build_records, list) or not isinstance(runtime_records, list):
        return False
    if build_environment.get("installed_distribution_count") != len(build_records):
        return False
    if runtime_environment.get("installed_distribution_count") != len(
        runtime_records
    ):
        return False

    def distribution_versions(records: list[Any]) -> dict[str, tuple[str, ...]] | None:
        versions: dict[str, list[str]] = {}
        for record in records:
            if not isinstance(record, dict):
                return None
            name = record.get("name")
            version = record.get("version")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
            ):
                return None
            versions.setdefault(name, []).append(version)
        return {
            name: tuple(sorted(package_versions))
            for name, package_versions in versions.items()
        }

    build_versions = distribution_versions(build_records)
    runtime_versions = distribution_versions(runtime_records)
    return (
        build_versions is not None
        and runtime_versions is not None
        and not WEBSHOP_REQUIREMENTS_LOCK_ISSUES
        and all(
            set(build_versions.get(name, ())) == {expected}
            and set(runtime_versions.get(name, ())) == {expected}
            for name, expected in WEBSHOP_RUNTIME_PACKAGE_VERSIONS.items()
        )
    )


def _webshop_index_validation(
    path: Path,
) -> tuple[list[str], dict[str, Any] | None, str | None]:
    """Validate and return the exact marker snapshot used by the checks."""

    issues: list[str] = []
    marker_payload: dict[str, Any] | None = None
    marker_sha256: str | None = None
    marker = path / WEBSHOP_FULL_INDEX_MARKER
    if marker.is_symlink() or not marker.is_file():
        issues.append(f"missing full-search-index marker {WEBSHOP_FULL_INDEX_MARKER}")
    else:
        try:
            raw_marker_payload, marker_sha256 = _read_webshop_full_index_marker(
                marker
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
            issues.append(f"invalid full-search-index marker: {exc}")
        else:
            if not isinstance(raw_marker_payload, dict):
                issues.append("invalid full-search-index marker: expected object")
            else:
                marker_payload = raw_marker_payload
                for key, expected in WEBSHOP_FULL_INDEX_PROFILE.items():
                    value = marker_payload.get(key)
                    if (
                        key in WEBSHOP_FULL_INDEX_INTEGER_FIELDS
                        and type(value) is not int
                    ) or value != expected:
                        issues.append(f"full-search-index marker mismatch for {key}")
    index_root = path / "search_engine" / "indexes"
    segment_files = (
        [
            candidate
            for candidate in index_root.glob("segments_*")
            if candidate.is_file()
        ]
        if index_root.is_dir()
        else []
    )
    if not segment_files or not all(
        candidate.stat().st_size > 0 for candidate in segment_files
    ):
        issues.append("missing full Lucene search index segments")
    if marker_payload is None:
        return issues, None, marker_sha256

    marker_index = marker_payload.get("index_content")
    if not _marker_digest_is_valid(marker_index, digest_key="root_sha256"):
        issues.append("full-search-index marker index-content root digest is invalid")
        return issues, marker_payload, marker_sha256
    try:
        current_index = webshop_index_content_provenance(path)
    except Exception as exc:
        issues.append(f"full Lucene committed-file manifest cannot be computed: {exc}")
        return issues, marker_payload, marker_sha256
    if marker_index.get("files") != current_index["files"]:
        issues.append("full Lucene committed-file SHA256 manifest mismatch")
    if marker_index.get("root_sha256") != current_index["root_sha256"]:
        issues.append("full Lucene committed-file root digest mismatch")

    marker_environment = marker_payload.get("build_environment")
    if not _marker_digest_is_valid(
        marker_environment,
        digest_key="identity_sha256",
    ):
        issues.append("full-search-index marker build-environment digest is invalid")
    else:
        try:
            current_environment = webshop_build_environment_provenance()
        except Exception as exc:
            issues.append(f"WebShop build environment cannot be fingerprinted: {exc}")
        else:
            if not _build_environment_is_runtime_compatible(
                marker_environment,
                current_environment,
            ):
                issues.append(
                    "WebShop build environment is incompatible with current runtime"
                )

    marker_source_data_index = marker_payload.get("source_data_index_identity")
    current_source_data_index = _source_data_index_identity(
        current_index["root_sha256"]
    )
    marker_source_data_index_root = marker_payload.get(
        "source_data_index_root_sha256"
    )
    if (
        not isinstance(marker_source_data_index, dict)
        or marker_source_data_index_root
        != _canonical_json_sha256(marker_source_data_index)
    ):
        issues.append("full-search-index marker source/data/index digest is invalid")
    elif marker_source_data_index != current_source_data_index:
        issues.append("full-search-index marker source/data/index identity mismatch")
    return issues, marker_payload, marker_sha256


def _webshop_index_issues(path: Path) -> list[str]:
    """Return marker, committed-file content, and environment issues."""

    issues, _marker_payload, _marker_sha256 = _webshop_index_validation(path)
    return issues


def webshop_dependency_issues(root: str | Path) -> list[str]:
    """Return provenance/data/index issues for the official full runtime."""

    path = Path(root).expanduser().resolve()
    return [
        *_webshop_source_data_issues(path),
        *_webshop_index_issues(path),
    ]


def _require_webshop_dependency_details(
    root: str | Path,
) -> tuple[Path, dict[str, Any], str, int]:
    path = Path(root).expanduser().resolve()
    index_issues, marker_payload, marker_sha256 = _webshop_index_validation(path)
    issues = [
        *_webshop_source_data_issues(path),
        *webshop_runtime_package_issues(),
        *webshop_java_runtime_issues(),
        *index_issues,
    ]
    indexed_documents: int | None = None
    if not issues:
        try:
            indexed_documents = webshop_index_document_count(path)
        except Exception as exc:
            issues.append(f"full Lucene search index cannot be opened: {exc}")
        else:
            if indexed_documents != WEBSHOP_FULL_INDEX_NUM_DOCUMENTS:
                issues.append(
                    "full Lucene search index document count mismatch: "
                    f"expected {WEBSHOP_FULL_INDEX_NUM_DOCUMENTS}, "
                    f"got {indexed_documents}"
                )
    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise RuntimeError(
            "official WebShop dependency verification failed:\n" + detail
        )
    if (
        marker_payload is None
        or marker_sha256 is None
        or indexed_documents is None
    ):
        raise RuntimeError(
            "official WebShop dependency verification failed to retain "
            "the validated marker identity"
        )
    return path, marker_payload, marker_sha256, indexed_documents


def require_webshop_dependency(root: str | Path) -> Path:
    """Return the verified root or fail before importing the official runtime."""

    path, _marker_payload, _marker_sha256, _indexed_documents = (
        _require_webshop_dependency_details(root)
    )
    return path


def webshop_runtime_package_issues() -> list[str]:
    """Return drift from the pinned compatible retrieval runtime."""

    issues: list[str] = list(WEBSHOP_REQUIREMENTS_LOCK_ISSUES)
    if list(sys.version_info[:2]) != WEBSHOP_RUNTIME_PYTHON:
        issues.append(
            "Python runtime mismatch: expected "
            f"{'.'.join(map(str, WEBSHOP_RUNTIME_PYTHON))}, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )
    for package, expected in WEBSHOP_RUNTIME_PACKAGE_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            issues.append(f"missing runtime package {package}=={expected}")
        else:
            if actual != expected:
                issues.append(
                    f"runtime package mismatch {package}: expected {expected}, got {actual}"
                )
    return issues


def webshop_java_runtime_issues(
    java_executable: str | Path | None = None,
) -> list[str]:
    """Return Java drift for the Pyserini/Lucene runtime."""

    java = str(java_executable or shutil.which("java") or "").strip()
    if not java:
        return ["Java runtime for Lucene is not found"]
    try:
        completed = subprocess.run(
            [java, "-version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return [f"Java runtime for Lucene cannot be executed: {exc}"]
    version_lines = (completed.stderr or completed.stdout).splitlines()
    first_line = version_lines[0] if version_lines else ""
    match = re.search(r'version "(?P<major>\d+)', first_line)
    if completed.returncode != 0 or not match:
        return [f"Java runtime version cannot be determined: {java}"]
    if int(match.group("major")) != WEBSHOP_RUNTIME_JAVA_MAJOR:
        return [
            f"Java runtime must be {WEBSHOP_RUNTIME_JAVA_MAJOR}; got {first_line}"
        ]
    return []


def webshop_environment_issues(
    root: str | Path,
    *,
    require_index: bool,
) -> list[str]:
    """Run the shared setup/preflight dependency contract."""

    path = Path(root).expanduser().resolve()
    issues = [
        *_webshop_source_data_issues(path),
        *webshop_runtime_package_issues(),
        *webshop_java_runtime_issues(),
    ]
    if not require_index:
        return issues

    index_issues = _webshop_index_issues(path)
    issues.extend(index_issues)
    if not index_issues:
        try:
            indexed_documents = webshop_index_document_count(path)
        except Exception as exc:
            issues.append(f"full Lucene search index cannot be opened: {exc}")
        else:
            if indexed_documents != WEBSHOP_FULL_INDEX_NUM_DOCUMENTS:
                issues.append(
                    "full Lucene search index document count mismatch: "
                    f"expected {WEBSHOP_FULL_INDEX_NUM_DOCUMENTS}, "
                    f"got {indexed_documents}"
                )
    return issues


def webshop_index_document_count(root: str | Path) -> int:
    """Open the actual Lucene index and return its live document count."""

    from pyserini.search.lucene import LuceneSearcher

    path = Path(root).expanduser().resolve() / "search_engine" / "indexes"
    searcher = LuceneSearcher(str(path))
    try:
        return int(searcher.num_docs)
    finally:
        close = getattr(searcher, "close", None)
        if callable(close):
            close()


def webshop_dependency_contract() -> dict[str, Any]:
    """Return the immutable declared WebShop dependency contract."""

    return {
        "source_repository": WEBSHOP_SOURCE_REPOSITORY,
        "source_commit": WEBSHOP_SOURCE_COMMIT,
        "source_checkout_root": "git_toplevel",
        "runtime_file_sha256": dict(WEBSHOP_RUNTIME_FILE_SHA256),
        "data_profile": "full",
        "data_source": WEBSHOP_DATA_SOURCE,
        "required_data_files": list(WEBSHOP_FULL_DATA_FILES),
        "data_file_sha256": dict(WEBSHOP_FULL_DATA_SHA256),
        "source_worktree_tracked_clean": True,
        "source_worktree_clean_including_untracked": True,
        "index_profile": "full",
        "index_marker": WEBSHOP_FULL_INDEX_MARKER,
        "index_marker_contract": dict(WEBSHOP_FULL_INDEX_PROFILE),
        "index_num_converted_documents": (
            WEBSHOP_FULL_INDEX_NUM_CONVERTED_DOCUMENTS
        ),
        "index_num_documents": WEBSHOP_FULL_INDEX_NUM_DOCUMENTS,
        "index_runtime_validation": (
            "rehash_latest_lucene_commit_and_match_live_document_count"
        ),
        "index_content_schema": WEBSHOP_INDEX_CONTENT_SCHEMA,
        "source_data_index_schema": WEBSHOP_SOURCE_DATA_INDEX_SCHEMA,
        "runtime_profile": WEBSHOP_RUNTIME_PROFILE,
        "runtime_profile_origin": "pinned_compatible_not_upstream_original",
        "build_environment_schema": WEBSHOP_BUILD_ENVIRONMENT_SCHEMA,
        "build_environment_validation": (
            "valid_build_manifest_and_compatible_python_java_runtime_dependencies"
        ),
        "python_major_minor": list(WEBSHOP_RUNTIME_PYTHON),
        "java_major": WEBSHOP_RUNTIME_JAVA_MAJOR,
        "requirements_lock": WEBSHOP_REQUIREMENTS_LOCK_RELATIVE,
        "requirements_lock_sha256": WEBSHOP_REQUIREMENTS_LOCK_SHA256,
        "runtime_package_versions": dict(WEBSHOP_RUNTIME_PACKAGE_VERSIONS),
        "environment_adapter": "official_text",
        "goal_profile": "official_human_test",
        "goal_split_definition": "baseline_models/env.py:test=range(500)",
        "goal_index_range": [
            WEBSHOP_OFFICIAL_TEST_GOAL_START,
            WEBSHOP_OFFICIAL_TEST_GOAL_END,
        ],
        "goal_shuffle_seed": 233,
        "human_goals": True,
        "num_products": None,
        "environment_instance_scope": "sequential_run_condition",
        "trial_reset": "fixed_integer_goal_index",
    }


def _require_observed_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(
            "official WebShop dependency verification failed: "
            f"validated marker has invalid {field}"
        )
    return value


def _require_observed_count(
    value: object,
    *,
    field: str,
    expected: int,
) -> int:
    if type(value) is not int or value != expected:
        raise RuntimeError(
            "official WebShop dependency verification failed: "
            f"validated marker has invalid {field}; expected {expected}, got {value!r}"
        )
    return value


def webshop_observed_dependency_contract(root: str | Path) -> dict[str, Any]:
    """Return the declared contract plus identity from one verified index marker."""

    _path, marker, marker_sha256, live_indexed_documents = (
        _require_webshop_dependency_details(root)
    )
    marker_sha256 = _require_observed_sha256(
        marker_sha256,
        field="marker_sha256",
    )

    index_content = marker.get("index_content")
    if not _marker_digest_is_valid(index_content, digest_key="root_sha256"):
        raise RuntimeError(
            "official WebShop dependency verification failed: "
            "validated marker index-content digest is invalid"
        )
    assert isinstance(index_content, dict)
    index_content_root_sha256 = _require_observed_sha256(
        index_content.get("root_sha256"),
        field="index_content_root_sha256",
    )

    source_data_index = marker.get("source_data_index_identity")
    source_data_index_root_sha256 = _require_observed_sha256(
        marker.get("source_data_index_root_sha256"),
        field="source_data_index_root_sha256",
    )
    if (
        not isinstance(source_data_index, dict)
        or source_data_index_root_sha256
        != _canonical_json_sha256(source_data_index)
        or source_data_index
        != _source_data_index_identity(index_content_root_sha256)
    ):
        raise RuntimeError(
            "official WebShop dependency verification failed: "
            "validated marker source/data/index identity is invalid"
        )

    build_environment = marker.get("build_environment")
    if not _marker_digest_is_valid(
        build_environment,
        digest_key="identity_sha256",
    ):
        raise RuntimeError(
            "official WebShop dependency verification failed: "
            "validated marker build-environment digest is invalid"
        )
    assert isinstance(build_environment, dict)
    build_environment_identity_sha256 = _require_observed_sha256(
        build_environment.get("identity_sha256"),
        field="build_environment_identity_sha256",
    )

    indexed_documents = _require_observed_count(
        marker.get("indexed_documents"),
        field="indexed_documents",
        expected=WEBSHOP_FULL_INDEX_NUM_DOCUMENTS,
    )
    if live_indexed_documents != indexed_documents:
        raise RuntimeError(
            "official WebShop dependency verification failed: "
            "validated marker indexed_documents does not match the live index"
        )
    converted_documents = _require_observed_count(
        marker.get("converted_documents"),
        field="converted_documents",
        expected=WEBSHOP_FULL_INDEX_NUM_CONVERTED_DOCUMENTS,
    )

    return {
        **webshop_dependency_contract(),
        "marker_sha256": marker_sha256,
        "index_content_root_sha256": index_content_root_sha256,
        "source_data_index_root_sha256": source_data_index_root_sha256,
        "build_environment_identity_sha256": (
            build_environment_identity_sha256
        ),
        "indexed_documents": indexed_documents,
        "converted_documents": converted_documents,
    }


def main() -> int:
    """Validate the same WebShop environment contract used by preflight."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("pre-index", "post-index"),
    )
    args = parser.parse_args()

    issues = webshop_environment_issues(
        args.root,
        require_index=args.phase == "post-index",
    )
    if issues:
        for issue in issues:
            print(f"[webshop-contract] {issue}", file=sys.stderr)
        return 2
    print(f"[webshop-contract] {args.phase} dependency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
