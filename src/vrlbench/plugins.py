"""Trusted component loading used by isolated simulation workers.

Importing this module does not import the reference runtime, alter registries,
or instantiate a model client. Plugins are executable Python, not a sandbox.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


_LOCAL_IMPORT_SOURCES: dict[str, Path] = {}


def _check_local_imports(path: Path, content: bytes) -> None:
    """Reject ambiguous absolute sibling imports without importing the plugin.

    Only statically declared imports that resolve to local Python sources are
    considered. Files unrelated to any declared local dependency are ignored.
    Package-relative imports retain Python's normal package isolation.
    """
    directory = path.parent
    pending = [(path, content)]
    inspected: set[Path] = set()
    sources: dict[str, Path] = {}
    while pending:
        source_path, source = pending.pop()
        if source_path in inspected:
            continue
        inspected.add(source_path)
        for node in ast.walk(ast.parse(source, filename=str(source_path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names
                                        if alias.name != "*")]
            else:
                continue
            for name in names:
                parts = name.split(".")
                for length in range(1, len(parts) + 1):
                    location = directory.joinpath(*parts[:length])
                    candidate = location.with_suffix(".py")
                    if not candidate.is_file():
                        candidate = location / "__init__.py"
                    if not candidate.is_file():
                        break
                    candidate = candidate.resolve()
                    if length == 1:
                        sources[parts[0]] = candidate
                    if candidate not in inspected:
                        pending.append((candidate, candidate.read_bytes()))
    for name, expected in sources.items():
        previous = _LOCAL_IMPORT_SOURCES.get(name)
        loaded = sys.modules.get(name)
        loaded_source = getattr(loaded, "__file__", None)
        if loaded is not None and (not loaded_source or Path(loaded_source).resolve() != expected):
            previous = Path(loaded_source).resolve() if loaded_source else Path("<built-in>")
        if previous is not None and previous != expected:
            raise ImportError(
                f"Conflicting local plugin import {name!r}: {expected} conflicts with {previous}; "
                "use distinct module names or installed packages with relative imports"
            )
    # A new search directory can also shadow another component's deferred
    # import even if this component never imports its own same-named file.
    for name, expected in _LOCAL_IMPORT_SOURCES.items():
        if name in sources or name in sys.modules:
            continue
        candidate = (directory / name).with_suffix(".py")
        if not candidate.is_file():
            candidate = directory / name / "__init__.py"
        if candidate.is_file() and candidate.resolve() != expected:
            raise ImportError(
                f"Conflicting local plugin import {name!r}: {candidate} would shadow "
                f"the deferred import from {expected}; use installed packages with relative imports"
            )
    # Reserve deferred imports too, before another component can shadow them.
    _LOCAL_IMPORT_SOURCES.update(sources)


def runtime_root() -> Path:
    """Locate the namespaced runtime shipped with the installed SDK."""
    root = Path(__file__).resolve().parent / "runtime"
    if not (root / "loop.py").is_file():
        raise FileNotFoundError("The VRL-Bench runtime is missing from this installation")
    return root


def reference_root() -> Path:
    """Deprecated compatibility name for :func:`runtime_root`."""
    return runtime_root()


def enable_reference_imports() -> Path:
    """Deprecated no-op: runtime imports now use ``vrlbench.runtime``."""
    return runtime_root()


def load_factory(spec: Mapping[str, Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Load ``/absolute/file.py:factory`` or ``module:factory``.

    File modules receive content-derived names rather than generic names such
    as ``strategy``. Their parent directory supports ordinary sibling imports;
    conflicting statically declared sibling sources fail instead of being reused.
    The caller must only invoke this function for trusted plugins.
    """
    entrypoint = spec.get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise ValueError("A plugin entrypoint must be file.py:factory or module:factory")
    module_name, attribute = entrypoint.rsplit(":", 1)
    if not module_name or not attribute or any(not part.isidentifier() for part in attribute.split(".")):
        raise ValueError(f"Invalid plugin entrypoint: {entrypoint!r}")
    kwargs = spec.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError("Plugin kwargs must be an object")
    identity: dict[str, Any] = {"entrypoint": entrypoint}
    if module_name.endswith(".py"):
        path = Path(module_name)
        if not path.is_absolute():
            raise ValueError("File plugin entrypoints must use an absolute path")
        path = path.resolve(strict=True)
        content = path.read_bytes()
        _check_local_imports(path, content)
        digest = hashlib.sha256(content).hexdigest()
        # A path suffix also distinguishes identical entry files in different directories.
        path_digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        unique_name = f"_vrlbench_plugin_{digest}_{path_digest}"
        module = sys.modules.get(unique_name)
        if module is None:
            module_spec = importlib.util.spec_from_file_location(unique_name, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Cannot load plugin {path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[unique_name] = module
            directory = str(path.parent)
            if directory in sys.path:
                sys.path.remove(directory)
            sys.path.insert(0, directory)
            try:
                # Execute the exact bytes whose hash identifies the candidate.
                # SourceFileLoader may reuse timestamp/size-based bytecode even
                # after a fast same-size edit of a memory-system plugin.
                code = compile(content, str(path), "exec", dont_inherit=True)
                exec(code, module.__dict__)
            except BaseException:
                sys.modules.pop(unique_name, None)
                raise
        identity.update({"source_path": str(path), "source_sha256": digest})
    else:
        module = importlib.import_module(module_name)
        source = getattr(module, "__file__", None)
        if source and Path(source).is_file():
            identity.update({"source_path": str(Path(source).resolve()),
                             "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()})
    factory: Any = module
    for part in attribute.split("."):
        factory = getattr(factory, part)
    if not callable(factory):
        raise TypeError(f"Plugin entrypoint is not callable: {entrypoint}")
    return factory, identity


def create_plugin(spec: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Instantiate a trusted factory using its explicit keyword arguments."""
    factory, identity = load_factory(spec)
    return factory(**dict(spec.get("kwargs", {}))), identity
