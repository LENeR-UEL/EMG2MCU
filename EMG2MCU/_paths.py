from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

_OUTPUT_ROOT: ContextVar[Path | None] = ContextVar("emg2mcu_output_root", default=None)


def output_root() -> Path:
    root = _OUTPUT_ROOT.get()
    return root if root is not None else Path.cwd() / "output"


@contextmanager
def output_context(root: str | Path | None) -> Iterator[None]:
    token = _OUTPUT_ROOT.set(Path(root) if root is not None else None)
    try:
        yield
    finally:
        _OUTPUT_ROOT.reset(token)
