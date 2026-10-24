"""Built-in task adapters for the shared trial-and-error runtime."""

from importlib import import_module

_REGISTRY = {
    "miniwob": "vrlbench.environments.miniwob",
    "webshop": "vrlbench.environments.webshop",
}


def get_benchmark(name: str):
    """Load a built-in environment adapter without importing unused stacks."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown workload {name!r}; choose from {', '.join(_REGISTRY)}")
    return import_module(_REGISTRY[key])


__all__ = ["get_benchmark"]
