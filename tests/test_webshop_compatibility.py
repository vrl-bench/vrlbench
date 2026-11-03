"""Bounded, offline checks for the WebShop native-runtime compatibility rules."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "webshop_dependency_under_test",
    ROOT / "src/vrlbench/environments/webshop_dependency.py",
)
dependency = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dependency)


def environment() -> dict:
    records = [
        {"name": name, "version": version}
        for name, version in dependency.WEBSHOP_RUNTIME_PACKAGE_VERSIONS.items()
    ]
    records.append({"name": "pip", "version": "26.1.2"})
    return {
        "schema_version": dependency.WEBSHOP_BUILD_ENVIRONMENT_SCHEMA,
        "runtime_profile": dependency.WEBSHOP_RUNTIME_PROFILE,
        "python": {
            "implementation": "CPython",
            "version": "3.10.20",
            "version_info": [3, 10, 20],
            "user_site_enabled": False,
        },
        "java": {"major": 11, "version": 'openjdk version "11.0.31"'},
        "requirements": {
            "kind": "pinned_compatible_runtime_constraints",
            "relative_path": dependency.WEBSHOP_REQUIREMENTS_LOCK_RELATIVE,
            "sha256": "old-comment-hash",
        },
        "installed_distribution_count": len(records),
        "installed_distributions": records,
    }


def signed_environment() -> dict:
    payload = environment()
    return {
        **payload,
        "identity_sha256": dependency._canonical_json_sha256(payload),
    }


class RuntimeCompatibility(unittest.TestCase):
    def test_packaged_lock_has_all_required_runtime_entries(self):
        self.assertEqual(dependency.WEBSHOP_REQUIREMENTS_LOCK_ISSUES, [])
        self.assertEqual(
            set(dependency.WEBSHOP_RUNTIME_PACKAGE_VERSIONS),
            dependency.WEBSHOP_RUNTIME_REQUIRED_DISTRIBUTIONS,
        )

    def test_patch_releases_tools_and_lock_comments_are_compatible(self):
        built = environment()
        current = deepcopy(built)
        current["python"].update(
            version="3.10.21", version_info=[3, 10, 21], user_site_enabled=True,
        )
        current["java"]["version"] = 'openjdk version "11.0.32.1"'
        current["requirements"]["sha256"] = "new-comment-hash"
        current["installed_distributions"][-1]["version"] = "23.0.1"
        current["installed_distributions"].append(
            {"name": "vrlbench", "version": "1.0.0"}
        )
        current["installed_distribution_count"] += 1
        self.assertTrue(
            dependency._build_environment_is_runtime_compatible(built, current)
        )

    def test_wrong_python_or_java_major_is_rejected(self):
        for field, update in (
            ("python", {"version_info": [3, 11, 0]}),
            ("java", {"major": 17}),
        ):
            with self.subTest(field=field):
                current = environment()
                current[field].update(update)
                self.assertFalse(
                    dependency._build_environment_is_runtime_compatible(
                        environment(), current,
                    )
                )

    def test_changed_or_missing_native_dependency_is_rejected(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                current = environment()
                records = current["installed_distributions"]
                record = next(row for row in records if row["name"] == "pyserini")
                if missing:
                    records.remove(record)
                    current["installed_distribution_count"] -= 1
                else:
                    record["version"] = "0.99.0"
                self.assertFalse(
                    dependency._build_environment_is_runtime_compatible(
                        environment(), current,
                    )
                )

    def test_inconsistent_distribution_count_is_rejected(self):
        current = environment()
        current["installed_distribution_count"] += 1
        self.assertFalse(
            dependency._build_environment_is_runtime_compatible(environment(), current)
        )

    def test_visible_inherited_distributions_are_resolved_in_import_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, inherited = root / "local", root / "inherited"
            for parent, package, version in (
                (local, "pip", "23.0.1"),
                (inherited, "pip", "26.1.2"),
                (inherited, "pyserini", "0.17.0"),
            ):
                info = parent / f"{package}-{version}.dist-info"
                info.mkdir(parents=True)
                (info / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
                )
            with patch.object(sys, "path", [str(local), str(inherited)]):
                records = dependency._installed_distribution_manifest()
            self.assertEqual(
                records,
                [
                    {"name": "pip", "version": "23.0.1"},
                    {"name": "pyserini", "version": "0.17.0"},
                ],
            )

    def test_independently_built_index_bytes_can_be_marked_and_validated(self):
        identities = []
        with tempfile.TemporaryDirectory() as directory:
            for number, content in enumerate((b"first-local-index", b"second-local-index")):
                with self.subTest(build=number):
                    root = Path(directory) / str(number)
                    index = root / "search_engine" / "indexes"
                    index.mkdir(parents=True)
                    (index / "segments_1").write_bytes(content)
                    with (
                        patch.object(dependency, "webshop_lucene_committed_file_names", return_value=("segments_1",)),
                        patch.object(dependency, "webshop_build_environment_provenance", side_effect=signed_environment),
                    ):
                        marker = dependency.write_webshop_full_index_marker(root)
                        issues, payload, digest = dependency._webshop_index_validation(root)
                    self.assertEqual(issues, [])
                    self.assertTrue(marker.is_file())
                    self.assertEqual(digest, hashlib.sha256(marker.read_bytes()).hexdigest())
                    self.assertEqual(payload["index_content"]["files"][0]["sha256"], hashlib.sha256(content).hexdigest())
                    identities.append(payload["source_data_index_root_sha256"])
        self.assertNotEqual(*identities)

    def test_changed_index_bytes_still_fail_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "search_engine" / "indexes"
            index.mkdir(parents=True)
            segment = index / "segments_1"
            segment.write_bytes(b"original-index")
            with (
                patch.object(dependency, "webshop_lucene_committed_file_names", return_value=("segments_1",)),
                patch.object(dependency, "webshop_build_environment_provenance", side_effect=signed_environment),
            ):
                dependency.write_webshop_full_index_marker(root)
                segment.write_bytes(b"changed-index")
                issues, _, _ = dependency._webshop_index_validation(root)
            self.assertIn("full Lucene committed-file SHA256 manifest mismatch", issues)
            self.assertIn("full Lucene committed-file root digest mismatch", issues)

    def test_marker_still_requires_the_declared_source_and_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "search_engine" / "indexes"
            index.mkdir(parents=True)
            (index / "segments_1").write_bytes(b"local-index")
            with (
                patch.object(dependency, "webshop_lucene_committed_file_names", return_value=("segments_1",)),
                patch.object(dependency, "webshop_build_environment_provenance", side_effect=signed_environment),
            ):
                marker = dependency.write_webshop_full_index_marker(root)
                original = json.loads(marker.read_text())
                for key, value in (("source_commit", "wrong-source"), ("data_file_sha256", {})):
                    with self.subTest(field=key):
                        marker.write_text(json.dumps({**original, key: value}))
                        issues, _, _ = dependency._webshop_index_validation(root)
                        self.assertIn(f"full-search-index marker mismatch for {key}", issues)

    def test_source_and_catalog_file_bytes_are_still_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".git").mkdir()
            source = root / "runtime.py"
            catalog = root / "catalog.json"
            source.write_bytes(b"source")
            catalog.write_bytes(b"catalog")

            def git_output(_root, *arguments):
                if arguments == ("rev-parse", "--show-toplevel"):
                    return 0, str(root)
                if arguments == ("rev-parse", "HEAD"):
                    return 0, dependency.WEBSHOP_SOURCE_COMMIT
                if arguments == ("remote", "get-url", "origin"):
                    return 0, dependency.WEBSHOP_SOURCE_REPOSITORY
                return 0, ""

            with (
                patch.object(dependency, "_git_output", side_effect=git_output),
                patch.object(dependency, "WEBSHOP_RUNTIME_FILE_SHA256", {"runtime.py": hashlib.sha256(b"source").hexdigest()}),
                patch.object(dependency, "WEBSHOP_FULL_DATA_FILES", ("catalog.json",)),
                patch.object(dependency, "WEBSHOP_FULL_DATA_SHA256", {"catalog.json": hashlib.sha256(b"catalog").hexdigest()}),
            ):
                self.assertEqual(dependency._webshop_source_data_issues(root), [])
                source.write_bytes(b"changed-source")
                catalog.write_bytes(b"changed-catalog")
                issues = dependency._webshop_source_data_issues(root)
            self.assertIn("runtime file hash mismatch runtime.py", issues)
            self.assertIn("full-data file hash mismatch catalog.json", issues)

    def test_corrupt_marker_digest_is_rejected(self):
        payload = environment()
        payload["identity_sha256"] = dependency._canonical_json_sha256(payload)
        self.assertTrue(
            dependency._marker_digest_is_valid(payload, digest_key="identity_sha256")
        )
        payload["java"]["major"] = 17
        self.assertFalse(
            dependency._marker_digest_is_valid(payload, digest_key="identity_sha256")
        )

    def test_live_index_document_count_remains_required(self):
        with (
            patch.object(dependency, "_webshop_index_validation", return_value=([], {}, "hash")),
            patch.object(dependency, "_webshop_source_data_issues", return_value=[]),
            patch.object(dependency, "webshop_runtime_package_issues", return_value=[]),
            patch.object(dependency, "webshop_java_runtime_issues", return_value=[]),
            patch.object(dependency, "webshop_index_document_count", return_value=3),
        ):
            with self.assertRaisesRegex(RuntimeError, "document count mismatch"):
                dependency._require_webshop_dependency_details(ROOT)


if __name__ == "__main__":
    unittest.main()
