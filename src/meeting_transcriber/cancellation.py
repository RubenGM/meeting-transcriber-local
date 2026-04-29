from __future__ import annotations

from typing import Callable


CancelCheck = Callable[[], bool]


class CancelledError(RuntimeError):
    pass


def never_cancelled() -> bool:
    return False


def raise_if_cancelled(cancelled: CancelCheck | None) -> None:
    if cancelled is not None and cancelled():
        raise CancelledError("Proceso cancelado por el usuario.")
