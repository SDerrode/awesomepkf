"""Linear smoothers ≡ RTS numerical equivalence check (linear-Gaussian pairwise).

On the linear-Gaussian pairwise model, the six linear smoothing variants (RTS,
BF, MBF, MF, DWY, VAR) are algebraically the *same* estimator (Geng et al., 2023):
each must agree with the RTS reference to machine precision. This script
quantifies, for every variant ``m`` in {BF, MBF, MF, DWY, VAR}, the residual gap
    eX = max_n || x_{n|N}^{m} - x_{n|N}^{RTS} ||_inf
    eP = max_n || P_{n|N}^{m} - P_{n|N}^{RTS} ||_inf
over a full smoothed trajectory, reporting the WORST case over several seeds for
three pairwise models. Emits a LaTeX table fragment for the report (§2.6,
``tab:smoothers-equiv``).

Run:  python -m prg.run_dwy_equivalence            # console table
      python -m prg.run_dwy_equivalence --out t.tex  # + LaTeX fragment
      python -m prg.run_dwy_equivalence --seeds 5 --N 200   # quicker
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prg.classes.linear_pks import (
    Linear_PKS_BF,
    Linear_PKS_DWY,
    Linear_PKS_MBF,
    Linear_PKS_MF,
    Linear_PKS_RTS,
    Linear_PKS_VAR,
)
from prg.classes.param_linear import ParamLinear
from prg.models.linear import ModelFactoryLinear

MODELS = {
    "x1y1": "model_x1_y1_AQ_pairwise",
    "x2y2": "model_x2_y2_AQ_pairwise",
    "x3y1": "model_x3_y1_AQ_pairwise",
}

# Variants compared against the RTS reference, in report-table column order.
VARIANTS = {
    "BF": Linear_PKS_BF,
    "MBF": Linear_PKS_MBF,
    "MF": Linear_PKS_MF,
    "DWY": Linear_PKS_DWY,
    "VAR": Linear_PKS_VAR,
}


def _param(model_name: str) -> ParamLinear:
    m = ModelFactoryLinear.create(model_name)
    pr = m.get_params().copy()
    dx = pr.pop("dim_x")
    dy = pr.pop("dim_y")
    return ParamLinear(0, dx, dy, **pr)


def _gaps_one_seed(model_name: str, seed: int, N: int) -> dict[str, tuple[float, float]]:
    """Per-variant worst per-step sup-norm gap vs RTS over one trajectory.

    The RTS reference is run once and reused across variants (same seed → same
    simulated trajectory), so the comparison is on identical data.
    """
    param = _param(model_name)
    rts = Linear_PKS_RTS(param, sKey=seed)
    rts.process_N_data_smoother(N=N)

    out: dict[str, tuple[float, float]] = {}
    for name, cls in VARIANTS.items():
        var = cls(param, sKey=seed)
        var.process_N_data_smoother(N=N)
        eX = eP = 0.0
        for a, b in zip(rts.history, var.history, strict=True):
            eX = max(eX, float(np.max(np.abs(
                np.asarray(a["Xkp1_smooth"], float)
                - np.asarray(b["Xkp1_smooth"], float)))))
            eP = max(eP, float(np.max(np.abs(
                np.asarray(a["PXXkp1_smooth"], float)
                - np.asarray(b["PXXkp1_smooth"], float)))))
        out[name] = (eX, eP)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", type=str, default=None, help="LaTeX table output path")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    print(f"Smoothers vs RTS equivalence — N+1={args.N + 1} steps, worst case over "
          f"{args.seeds} seeds\n")

    # rows[tag][variant] = (worst eX, worst eP)
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    header = f"{'model':<7}" + "".join(f"{v:>24}" for v in VARIANTS)
    print(header)
    print(f"{'':7}" + "".join(f"{'eX / eP':>24}" for _ in VARIANTS))
    print("-" * len(header))
    for tag, name in MODELS.items():
        worst = dict.fromkeys(VARIANTS, (0.0, 0.0))
        for s in seeds:
            gaps = _gaps_one_seed(name, s, args.N)
            worst = {
                v: (max(worst[v][0], gaps[v][0]), max(worst[v][1], gaps[v][1]))
                for v in VARIANTS
            }
        rows[tag] = worst
        cells = "".join(f"{f'{worst[v][0]:.1e} / {worst[v][1]:.1e}':>24}" for v in VARIANTS)
        print(f"{tag:<7}{cells}")

    if args.out:

        def _texp(v: float) -> str:
            """Format as a clean 10^{exp} power for LaTeX (French decimals)."""
            if v == 0.0:
                return "0"
            exp = int(np.floor(np.log10(v)))
            mant = v / 10.0**exp
            return f"{mant:.1f}".replace(".", "{,}") + r"\times 10^{" + f"{exp}" + "}"

        # Emits the data rows of §2.6 tab:smoothers-equiv (columns BF/MBF/MF/DWY/VAR,
        # two rows e_x / e_P per model). Paste between \midrule and \bottomrule.
        lines: list[str] = []
        for i, tag in enumerate(MODELS):
            if i:
                lines.append(r"        \addlinespace[2pt]")
            wx = " & ".join(f"${_texp(rows[tag][v][0])}$" for v in VARIANTS)
            wp = " & ".join(f"${_texp(rows[tag][v][1])}$" for v in VARIANTS)
            lines.append(f"        \\texttt{{{tag}}} & $e_{{\\ovx}}$ & {wx} \\\\")
            lines.append(f"        \\texttt{{{tag}}} & $e_{{\\mP}}$  & {wp} \\\\")
        out = Path(args.out)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nLaTeX table rows written to {out}")


if __name__ == "__main__":
    main()
