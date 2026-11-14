"""Read-only preflight checks for a configured evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .simulation import plan
from .specs import SimulationSpec


def diagnose(spec: SimulationSpec) -> dict:
    """Validate prerequisites without loading an agent or starting an environment."""
    spec = spec.resolved()
    plan(spec)
    checks = []

    def check(name: str, ready: bool, detail: str) -> None:
        checks.append({"name": name, "ready": ready, "detail": detail})

    online = spec.workload.factory is None or spec.workload.params.get("uses_model") is True
    visual = spec.workload.name == "osworld" or spec.workload.params.get("observation_mode") == "screenshot" or bool(
        spec.workload.params.get("include_screenshot", False)
        or spec.workload.params.get("requires_vision", False))
    for package in ("openai", "packaging"):
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        check(package, installed is not None,
              installed or "Install vrlbench[runtime].")
    if online:
        check("model", bool(spec.model.name) and "YOUR_" not in spec.model.name,
              "Choose an actual model name" if "YOUR_" in spec.model.name else spec.model.name)
        if spec.model.transport is None:
            check("credential", bool(os.environ.get(spec.model.api_key_env)),
                  f"Set {spec.model.api_key_env} in the calling shell.")
        if spec.actor.name == "openhands":
            check("simulator_python", sys.version_info >= (3, 12),
                  "OpenHands requires Python 3.12+.")
            try:
                sdk = importlib.metadata.version("openhands-sdk")
            except importlib.metadata.PackageNotFoundError:
                sdk = None
            check("openhands", sdk is not None, sdk or "Install vrlbench[openhands].")

    modules = {"miniwob": "synapse", "webshop": "web_agent_site",
               "webarena": "webarena", "osworld": "desktop_env", "browser_smoke": "playwright"}
    if online and spec.workload.name in modules:
        interpreter = spec.workload.params.get("environment_python", sys.executable)
        module = modules[spec.workload.name]
        root_variable = {"miniwob": "SYNAPSE_ROOT", "webshop": "WEBSHOP_ROOT"}.get(spec.workload.name)
        probe = "import importlib.util,json,os,sys\nfrom pathlib import Path\n"
        if root_variable:
            probe += f"root=os.environ.get({root_variable!r})\nif root: sys.path.insert(0,root)\n"
        probe += (f"info={{'module_available':importlib.util.find_spec({module!r}) is not None,"
                  "'python':list(sys.version_info[:3])}\n")
        if spec.workload.name == "browser_smoke":
            probe += ("if info['module_available']:\n"
                      " from playwright.sync_api import sync_playwright\n"
                      " with sync_playwright() as browser:\n"
                      "  info['browser_available']=Path(browser.chromium.executable_path).is_file()\n")
        if spec.workload.name == "miniwob":
            probe += ("if info['module_available']:\n"
                      " from vrlbench.environments.synapse_dependency import configure_synapse_import_path\n"
                      " configure_synapse_import_path()\n"
                      " info['browser_available']=all(Path(os.environ.get(k,'')).is_file() for k in ('SE_CHROME','SE_CHROMEDRIVER'))\n")
        probe += "print(json.dumps(info))"
        try:
            from .environments.import_path import isolated_package_path
            with isolated_package_path() as import_root:
                result = subprocess.run([interpreter, "-c", probe], capture_output=True,
                                        text=True, timeout=15, check=True,
                                        env={**os.environ, **spec.runtime_env, "PYTHONPATH": import_root})
            info = json.loads(result.stdout)
            check("environment", info["module_available"],
                  f"{module} in {interpreter}; Python {'.'.join(map(str, info['python']))}")
            if spec.workload.name in {"browser_smoke", "miniwob"}:
                check("browser_binary", info.get("browser_available", False),
                      "Use vrlbench setup miniwob or set SE_CHROME/SE_CHROMEDRIVER."
                      if spec.workload.name == "miniwob" else
                      "Install Chromium with python -m playwright install chromium.")
            if spec.workload.name == "miniwob":
                root = Path(spec.runtime_env.get("SYNAPSE_ROOT", os.environ.get("SYNAPSE_ROOT", "")))
                for case in spec.workload.cases:
                    task = root / "synapse/envs/miniwob/html/miniwob" / (case["task"] + ".html")
                    check("task_file", task.is_file(), str(task))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            check("environment", False,
                  f"Cannot probe {module} in {interpreter}. Follow the workload setup guide.")
        for case in spec.workload.cases:
            if isinstance(case, dict) and "config_file" in case:
                path = Path(case["config_file"])
                check("task_file", path.is_file(), str(path))
        if spec.workload.name == "osworld":
            vm = Path(spec.workload.params["env_kwargs"]["path_to_vm"])
            check("vm", vm.exists(), str(vm))

    output = Path(spec.output_dir)
    available = not output.exists() or (output.is_dir() and not any(output.iterdir()))
    check("output_directory", available, "Use a new or empty run directory.")
    return {"status": "ready" if all(c["ready"] for c in checks) else "blocked",
            "workload": spec.workload.name, "actor": spec.actor.name,
            "requires_vision": visual, "checks": checks,
            "model_calls_executed": 0,
            "scope": "Local prerequisites only. Model vision/tool support and live environment resets require an online smoke run."}
