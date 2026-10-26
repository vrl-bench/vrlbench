"""SDK extras must coexist with the supported native environment locks."""

from pathlib import Path
from importlib.metadata import PackageNotFoundError, distribution
import unittest
from urllib.parse import unquote, urlparse

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development environments.
    import tomli as tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "src" / "vrlbench" / "requirements"


def locked_versions(filename):
    versions = {}
    for raw in (LOCKS / filename).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        if requirement.url:
            wheel = Path(unquote(urlparse(requirement.url).path)).name
            _name, version, _build, _tags = parse_wheel_filename(wheel)
            version = str(version)
        else:
            specifiers = list(requirement.specifier)
            if len(specifiers) != 1 or specifiers[0].operator != "==":
                raise ValueError(f"expected one exact version in {filename}: {line}")
            version = specifiers[0].version
        versions[canonicalize_name(requirement.name)] = version
    return versions


class DependencyConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"]["optional-dependencies"]

    def assert_extra_fits(self, extra, filename, python_version):
        versions = locked_versions(filename)
        environment = default_environment()
        environment.update(python_version=python_version, python_full_version=python_version + ".0",
                           sys_platform="linux", platform_system="Linux", platform_machine="x86_64",
                           extra=extra)
        for raw in self.extras[extra]:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            name = canonicalize_name(requirement.name)
            with self.subTest(extra=extra, lock=filename, requirement=raw):
                self.assertIn(name, versions, f"{filename} must include this SDK dependency")
                self.assertTrue(requirement.specifier.contains(versions[name]),
                                f"{raw} conflicts with {name}=={versions[name]} in {filename}")

    def test_runtime_extra_fits_both_native_environments(self):
        self.assert_extra_fits("runtime", "miniwob-py312.lock.txt", "3.12")
        self.assert_extra_fits("runtime", "webshop-py310.lock.txt", "3.10")

    def test_analysis_extra_fits_webshop_numpy_and_scipy(self):
        self.assert_extra_fits("analysis", "webshop-py310.lock.txt", "3.10")

    def test_installed_openai_dependencies_fit_webshop_lock(self):
        try:
            installed = distribution("openai")
        except PackageNotFoundError:
            self.skipTest("requires the optional runtime dependencies")
        runtime = next(Requirement(raw) for raw in self.extras["runtime"]
                       if canonicalize_name(Requirement(raw).name) == "openai")
        if not runtime.specifier.contains(installed.version):
            self.skipTest("installed OpenAI version differs from the source runtime requirement")
        versions = locked_versions("webshop-py310.lock.txt")
        environment = default_environment()
        environment.update(python_version="3.10", python_full_version="3.10.0", extra="")
        for raw in installed.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            name = canonicalize_name(requirement.name)
            with self.subTest(requirement=raw):
                self.assertIn(name, versions)
                self.assertTrue(requirement.specifier.contains(versions[name]),
                                f"{raw} conflicts with WebShop's {name}=={versions[name]}")

    def test_full_webshop_lock_retains_official_retrieval_versions(self):
        full = locked_versions("webshop-py310.lock.txt")
        official = locked_versions("webshop-official-py310.lock.txt")
        for name, version in official.items():
            with self.subTest(package=name):
                self.assertEqual(full[name], version)


if __name__ == "__main__":
    unittest.main()
