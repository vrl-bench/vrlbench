#!/usr/bin/env python3
"""Exercise the supplied seed's memory lifecycle, without a model or environment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vrlbench.memory import MemorySystem  # noqa: E402


def load_seed() -> MemorySystem:
    path = Path(__file__).with_name("initial_strategy.py")
    spec = importlib.util.spec_from_file_location("vrlbench_memory_seed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    memory = module.create_memory()
    if not isinstance(memory, MemorySystem):
        raise TypeError("create_memory() must return an MemorySystem")
    return memory


def main() -> None:
    memory = load_seed()
    memory.reset()
    before = memory.read_for_trial(trial_index=0, total_trials=2, model="synthetic")
    assert before == ""
    trace = "Synthetic trial: search -> wrong item -> unsuccessful outcome"
    memory.on_trial_end(trace, success=False, model="synthetic", trial_index=0,
                        total_trials=2, metadata={"has_future_trial": True})
    after = memory.read_for_trial(trial_index=1, total_trials=2, model="synthetic")
    assert trace in after
    memory.reset()
    assert memory.build_section() == ""
    print(json.dumps({"interface_check": "passed", "write_read_reset": True,
                      "task_performance_evaluated": False}, sort_keys=True))


if __name__ == "__main__":
    main()
