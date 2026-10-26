"""Check standard Python distributions against the simulator source tree."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import tarfile
import unittest
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "vrlbench"


def package_sources() -> list[Path]:
    return sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)


def runtime_resources() -> list[Path]:
    prompts = PACKAGE / "runtime" / "er" / "source_prompts"
    return sorted({
        *(path for path in prompts.iterdir() if path.is_file()),
        *(PACKAGE / "requirements").glob("*.txt"),
        PACKAGE / "_replay.html",
        PACKAGE / "environments" / "synapse_overlay" / "LICENSE",
    })


def assert_public_paths(test: unittest.TestCase, names: list[str]) -> None:
    for name in names:
        parts = PurePosixPath(name).parts
        with test.subTest(path=name):
            test.assertFalse(PurePosixPath(name).is_absolute())
            test.assertNotIn("..", parts)
            test.assertFalse(set(parts) & {".git", "__pycache__", "experiments", "reproduce",
                                           "_paper", "_reference", "archive", "private"})
            test.assertNotIn(PurePosixPath(name).name,
                             {"SOURCE_MANIFEST.json", "MANIFEST.json", ".env", "id_rsa", "id_ed25519"})


class PackageLayout(unittest.TestCase):
    def test_runtime_prompts_licenses_and_environment_resources_are_available(self):
        required = [
            "runtime/loop.py", "runtime/er/reflexion.py", "runtime/er/ace.py",
            "runtime/er/dynamic_cheatsheet.py", "runtime/controllers/dsst.py",
            "runtime/er/source_prompts/REFLEXION_LICENSE",
            "runtime/er/source_prompts/DYNAMIC_CHEATSHEET_LICENSE",
            "runtime/er/source_prompts/ACE_LICENSE",
            "runtime/er/source_prompts/THIRD_PARTY_NOTICES",
            "environments/miniwob.py", "environments/webshop.py",
            "environments/synapse_overlay/synapse/envs/miniwob/environment.py",
            "environments/synapse_overlay/LICENSE",
            "requirements/miniwob-py312.lock.txt", "requirements/webshop-official-py310.lock.txt",
        ]
        for name in required:
            with self.subTest(resource=name):
                self.assertTrue((PACKAGE / name).is_file())
        self.assertTrue(package_sources())
        self.assertTrue(runtime_resources())
        self.assertFalse((ROOT / "vrlbench.py").exists())
        self.assertFalse((ROOT / "examples/memory_strategy/requirements.txt").exists())


class BuiltDistributions(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("VRLBENCH_TEST_WHEEL"), "build a wheel to check installed resources")
    def test_wheel_contains_simulator_sources_and_resources(self):
        with ZipFile(Path(os.environ["VRLBENCH_TEST_WHEEL"])) as archive:
            names = archive.namelist()
            self.assertEqual(len(names), len(set(names)))
            assert_public_paths(self, names)
            self.assertFalse(any(name.startswith(("vex2/", "benchmarks/")) for name in names))
            for path in [*package_sources(), *runtime_resources()]:
                installed = "vrlbench/" + path.relative_to(PACKAGE).as_posix()
                with self.subTest(resource=installed):
                    self.assertEqual(archive.read(installed), path.read_bytes())
            entries = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
            self.assertIn("vrlbench = vrlbench.cli:main", archive.read(entries).decode())
            for name in ("LICENSE", "THIRD_PARTY_NOTICES"):
                self.assertTrue(any(path.endswith("/licenses/" + name) for path in names), name)

    @unittest.skipUnless(os.environ.get("VRLBENCH_TEST_SDIST"), "build an sdist to check source inclusion")
    def test_sdist_contains_full_simulator_and_examples(self):
        with tarfile.open(Path(os.environ["VRLBENCH_TEST_SDIST"]), "r:gz") as archive:
            files = [member.name for member in archive.getmembers() if member.isfile()]
            root = files[0].split("/")[0]
            relative = [name.removeprefix(root + "/") for name in files]
            self.assertEqual(len(relative), len(set(relative)))
            assert_public_paths(self, relative)
            self.assertNotIn("vrlbench.py", relative)
            self.assertNotIn("examples/memory_strategy/requirements.txt", relative)
            sources = [*package_sources(), *runtime_resources()]
            sources.extend(ROOT / name for name in (
                "pyproject.toml", "MANIFEST.in", "README.md", "README.zh-CN.md",
                "LICENSE", "THIRD_PARTY_NOTICES",
            ))
            sources.extend((ROOT / "examples").rglob("*.py"))
            sources.extend((ROOT / "examples").rglob("*.json"))
            sources.extend((ROOT / "tests").glob("*.py"))
            sources.extend((ROOT / "docs").glob("*.md"))
            sources.extend((ROOT / "scripts").glob("*.py"))
            sources.extend((ROOT / "scripts").glob("*.sh"))
            for pattern in ("*.md", "*.svg", "*.png"):
                sources.extend((ROOT / "docs").rglob(pattern))
            for path in sources:
                name = path.relative_to(ROOT).as_posix()
                with self.subTest(source=name):
                    stream = archive.extractfile(root + "/" + name)
                    self.assertIsNotNone(stream)
                    self.assertEqual(stream.read(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
