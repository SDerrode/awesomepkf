"""CheckResult / DiagnosticReport — output containers."""

from dataclasses import dataclass, field

from prg.classes.matrix_diagnostics.status import Status

__all__ = ["CheckResult", "DiagnosticReport"]


@dataclass
class CheckResult:
    name: str
    status: Status
    value: float | None
    threshold: float | None
    message: str

    def __str__(self) -> str:
        thr = f"  (threshold: {self.threshold})" if self.threshold is not None else ""
        val = f"  [value: {self.value:.6g}]" if self.value is not None else ""
        return f"  {self.status!s:<14}  {self.name}{val}{thr}\n    → {self.message}"


@dataclass
class DiagnosticReport:
    matrix_type: str
    shape: tuple
    dtype: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_status(self) -> Status:
        if any(c.status == Status.FAIL for c in self.checks):
            return Status.FAIL
        if any(c.status == Status.WARNING for c in self.checks):
            return Status.WARNING
        return Status.OK

    @property
    def is_ok(self) -> bool:
        """True only if all checks are OK (no warnings, no failures)."""
        return self.overall_status == Status.OK

    @property
    def is_valid(self) -> bool:
        """True if no check is FAIL (warnings tolerated)."""
        return self.overall_status != Status.FAIL

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Diagnostic — {self.matrix_type}",
            f"  Shape : {self.shape}    dtype : {self.dtype}",
            f"  Overall : {self.overall_status}",
            "-" * 60,
        ]
        lines.extend(str(check) for check in self.checks)
        lines.append("=" * 60)
        return "\n".join(lines)
