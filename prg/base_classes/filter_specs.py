"""
Filter specifications registry.

Each entry binds a short filter name (used in CLI, history filenames,
plot basenames) to:
- the actual filter class to instantiate;
- a display acronym;
- a flag indicating whether the filter is linear (PKF) or nonlinear;
- the names of any extra constructor kwargs the filter requires
  (e.g. ``sigmaSet`` for UKF/UPKF, ``n_particles`` for PF/PPF).

Adding a new filter is a single dict entry — no new runner class needed.
"""

from dataclasses import dataclass, field

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.nonlinear_epkf import NonLinear_EPKF
from prg.classes.nonlinear_pf import NonLinear_PF
from prg.classes.nonlinear_ppf import NonLinear_PPF
from prg.classes.nonlinear_ukf import NonLinear_UKF
from prg.classes.nonlinear_upkf import NonLinear_UPKF

__all__ = ["FILTER_SPECS", "FilterSpec"]


@dataclass(frozen=True)
class FilterSpec:
    """Specification used by :class:`FilterRunner` to instantiate a filter."""

    name: str
    acronym: str
    filter_class: type
    is_linear: bool = False
    requires: tuple[str, ...] = field(default_factory=tuple)


FILTER_SPECS: dict[str, FilterSpec] = {
    "pkf":  FilterSpec("pkf",  "PKF",  Linear_PKF,     is_linear=True),
    "epkf": FilterSpec("epkf", "EPKF", NonLinear_EPKF),
    "upkf": FilterSpec("upkf", "UPKF", NonLinear_UPKF, requires=("sigmaSet",)),
    "ukf":  FilterSpec("ukf",  "UKF",  NonLinear_UKF,  requires=("sigmaSet",)),
    "pf":   FilterSpec("pf",   "PF",   NonLinear_PF,   requires=("n_particles",)),
    "ppf":  FilterSpec("ppf",  "PPF",  NonLinear_PPF,  requires=("n_particles",)),
}
