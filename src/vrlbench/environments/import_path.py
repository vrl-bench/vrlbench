"""Expose only the pure-Python harness to another interpreter, not its dependencies."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def isolated_package_path():
    # A wheel lives beside all actor dependencies in site-packages. Putting
    # that parent on PYTHONPATH breaks environment locks and Python 3.10/3.12
    # isolation. A package-only link keeps resources available without leaking
    # the adjacent interpreter's extensions or third-party modules.
    with TemporaryDirectory(prefix='vrlbench-env-import-') as directory:
        Path(directory, 'vrlbench').symlink_to(Path(__file__).resolve().parents[1],
                                              target_is_directory=True)
        yield directory
