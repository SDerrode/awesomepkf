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

    @staticmethod
    def _format_float_or_none(value: float | None, label: str) -> str:
        if value is None:
            return ""
        return f"  [{label}: {value:.6g}]"

    def __str__(self) -> str:
        thr = self._format_float_or_none(self.threshold, "threshold")
        val = self._format_float_or_none(self.value, "value")
        return f"  {self.status!s:<14}  {self.name}{val}{thr}\n    → {self.message}"


@dataclass
class DiagnosticReport:
    matrix_type: str
    shape: tuple
    dtype: str
    checks: list[CheckResult] = field(default_factory=list)

    def _aggregate_status(self) -> tuple[bool, bool]:
        has_warning = False
        has_fail = False
        for check in self.checks:
            if check.status == Status.FAIL:
                has_fail = True
                break
            if check.status == Status.WARNING:
                has_warning = True
        return has_warning, has_fail

    @property
    def overall_status(self) -> Status:
        has_warning, has_fail = self._aggregate_status()
        if has_fail:
            return Status.FAIL
        if has_warning:
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
