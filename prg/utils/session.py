"""Session config (TOML) — filter + model + parameter overrides.

Tracked, PyQt-free home for the session-config serialisation used by the
``--config`` CLI flag (:mod:`prg.run_filter`) and by the GUI. Kept out of the
gitignored ``prg.gui`` package so the shipped ``--config`` feature works on a
plain ``pip install awesomepkf`` (the GUI re-exports these names for backward
compatibility via :mod:`prg.gui.session`).

TOML schema
-----------

::

    label = "EPKF on retroactions"

    [filter]
    name        = "epkf"        # one of: pkf, epkf, upkf, ukf, pf, ppf
    sigma_set   = "wan2000"     # ukf / upkf only  (optional)
    n_particles = 500           # pf / ppf only    (optional)
    smoother    = false         # also run the matching smoother (optional)
    joseph      = false         # Joseph form for the Kalman update (optional)

    [model]
    name = "model_x1_y1_pairwise"

    [model.overrides]           # optional: scalar attributes set on the param
    q_x = 0.04

Reading uses the standard-library ``tomllib`` (Python 3.11+), falling back to
``tomli`` on 3.10. Writing uses ``tomli_w`` (imported lazily, so merely reading
a config — the CLI ``--config`` path — never requires the writer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = ["SessionConfig", "load_config", "save_config"]


@dataclass(frozen=True)
class SessionConfig:
    """A serialisable description of a filter + model setup."""

    filter_name: str
    model_name: str
    sigma_set: str | None = None
    n_particles: int | None = None
    overrides: dict[str, float] = field(default_factory=dict)
    label: str = ""
    smoother: bool = False
    joseph: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        from prg.base_classes.filter_specs import FILTER_SPECS

        if self.filter_name not in FILTER_SPECS:
            raise ValueError(
                f"Unknown filter {self.filter_name!r}. "
                f"Expected one of {sorted(FILTER_SPECS)}.",
            )
        if self.n_particles is not None and self.n_particles <= 0:
            raise ValueError(
                f"n_particles must be > 0, got {self.n_particles!r}.",
            )
        for key, value in self.overrides.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Override {key!r} must be a number, got {type(value).__name__}.",
                )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"filter": {"name": self.filter_name}}
        if self.sigma_set is not None:
            out["filter"]["sigma_set"] = self.sigma_set
        if self.n_particles is not None:
            out["filter"]["n_particles"] = int(self.n_particles)
        if self.smoother:
            out["filter"]["smoother"] = True
        if self.joseph:
            out["filter"]["joseph"] = True
        out["model"] = {"name": self.model_name}
        if self.overrides:
            out["model"]["overrides"] = {
                k: float(v) for k, v in self.overrides.items()
            }
        if self.label:
            out["label"] = self.label
        return out

    def save(self, path: str | Path) -> None:
        save_config(self, Path(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionConfig:
        flt = data.get("filter") or {}
        mdl = data.get("model") or {}
        if "name" not in flt:
            raise ValueError("Missing required key: filter.name")
        if "name" not in mdl:
            raise ValueError("Missing required key: model.name")
        return cls(
            filter_name=str(flt["name"]),
            model_name=str(mdl["name"]),
            sigma_set=flt.get("sigma_set"),
            n_particles=flt.get("n_particles"),
            overrides=dict(mdl.get("overrides") or {}),
            label=str(data.get("label", "")),
            smoother=bool(flt.get("smoother", False)),
            joseph=bool(flt.get("joseph", False)),
        )


def load_config(path: str | Path) -> SessionConfig:
    """Read a TOML file and return a :class:`SessionConfig`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such config file: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return SessionConfig.from_dict(data)


def save_config(cfg: SessionConfig, path: str | Path) -> None:
    """Write a :class:`SessionConfig` to a TOML file (atomically)."""
    try:
        import tomli_w
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Writing a session config requires 'tomli-w' (pip install tomli-w).",
        ) from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        tomli_w.dump(cfg.to_dict(), fh)
    tmp.replace(path)
