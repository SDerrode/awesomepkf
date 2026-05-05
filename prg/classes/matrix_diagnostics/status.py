"""Status enum used in CheckResult / DiagnosticReport."""

from enum import Enum, auto

__all__ = ["Status"]


class Status(Enum):
    OK = auto()
    WARNING = auto()
    FAIL = auto()

    def __str__(self) -> str:
        return {
            Status.OK: "✅ OK",
            Status.WARNING: "⚠️  WARNING",
            Status.FAIL: "❌ FAIL",
        }[self]
