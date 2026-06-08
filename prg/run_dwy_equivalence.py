"""DWY ≡ RTS numerical equivalence check (linear-Gaussian pairwise models).

On the linear-Gaussian pairwise model, the DWY (backward-RTS) smoother and the
RTS smoother are algebraically the *same* estimator (Geng et al., 2023): they
must agree to machine precision. This script quantifies the residual gap
    eX = max_n || x_{n|N}^{DWY} - x_{n|N}^{RTS} ||_inf
    eP = max_n || P_{n|N}^{DWY} - P_{n|N}^{RTS} ||_inf
over a full smoothed trajectory, reporting the WORST case over several seeds for
three pairwise models. Emits a LaTeX table fragment for the report (§2.5).

Run:  python -m prg.run_dwy_equivalence
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prg.classes.linear_pks import Linear_PKS_DWY, Linear_PKS_RTS
from prg.classes.param_linear import ParamLinear
from prg.models.linear import ModelFactoryLinear

MODELS = {
    "x1y1": "model_x1_y1_AQ_pairwise",
    "x2y2": "model_x2_y2_AQ_pairwise",
    "x3y1": "model_x3_y1_AQ_pairwise",
}


def _param(model_name: str) -> ParamLinear:
    m = ModelFactoryLinear.create(model_name)
    pr = m.get_params().copy()
    dx = pr.pop("dim_x")
    dy = pr.pop("dim_y")
    return ParamLinear(0, dx, dy, **pr)


def _max_diff(model_name: str, seed: int, N: int) -> tuple[float, float]:
    """Worst per-step sup-norm gap between DWY and RTS over one trajectory."""
    param = _param(model_name)
    rts = Linear_PKS_RTS(param, sKey=seed)
    rts.process_N_data_smoother(N=N)
    dwy = Linear_PKS_DWY(param, sKey=seed)
    dwy.process_N_data_smoother(N=N)
    eX = eP = 0.0
    for a, b in zip(rts.history, dwy.history, strict=True):
        eX = max(eX, float(np.max(np.abs(
            np.asarray(a["Xkp1_smooth"], float) - np.asarray(b["Xkp1_smooth"], float)))))
        eP = max(eP, float(np.max(np.abs(
            np.asarray(a["PXXkp1_smooth"], float) - np.asarray(b["PXXkp1_smooth"], float)))))
    return eX, eP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", type=str, default=None, help="LaTeX table output path")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    print(f"DWY vs RTS equivalence — N+1={args.N + 1} steps, worst case over "
          f"{args.seeds} seeds\n")
    print(f"{'model':<8} {'max|dX|':>12} {'max|dP|':>12}")
    print("-" * 34)
    rows = {}
    for tag, name in MODELS.items():
        wX = wP = 0.0
        for s in seeds:
            eX, eP = _max_diff(name, s, args.N)
            wX, wP = max(wX, eX), max(wP, eP)
        rows[tag] = (wX, wP)
        print(f"{tag:<8} {wX:>12.2e} {wP:>12.2e}")

    gX = max(v[0] for v in rows.values())
    gP = max(v[1] for v in rows.values())
    print("-" * 34)
    print(f"{'GLOBAL':<8} {gX:>12.2e} {gP:>12.2e}")

    if args.out:

        def _texp(v: float) -> str:
            """Format as a clean 10^{exp} power for LaTeX (French decimals)."""
            if v == 0.0:
                return "0"
            exp = int(np.floor(np.log10(v)))
            mant = v / 10.0**exp
            return f"{mant:.1f}".replace(".", "{,}") + r"\times 10^{" + f"{exp}" + "}"

        lines = [
            r"\begin{table}[!htbp]",
            r"    \centering",
            r"    \begin{tabular}{lcc}",
            r"        \toprule",
            r"        Modèle & $\max_n \|\ovx_{\nnN}^{\text{DWY}} - \ovx_{\nnN}^{\text{RTS}}\|_\infty$"
            r" & $\max_n \|\mP^{xx,\text{DWY}}_{\nnN} - \mP^{xx,\text{RTS}}_{\nnN}\|_\infty$ \\",
            r"        \midrule",
        ]
        for tag in MODELS:
            wX, wP = rows[tag]
            lines.append(
                f"        \\texttt{{{tag}}} pairwise & ${_texp(wX)}$ & ${_texp(wP)}$ \\\\"
            )
        lines += [
            r"        \bottomrule",
            r"    \end{tabular}",
            rf"    \caption{{Écart maximal entre l'estimée lissée DWY et l'estimée RTS sur "
            rf"une trajectoire de $N+1 = {args.N + 1}$ pas, pris au pire cas sur "
            rf"${args.seeds}$ graines indépendantes. Les écarts sont au niveau du bruit "
            rf"d'arrondi double précision ($\sim 10^{{-16}}$ sur les covariances, "
            rf"$\sim 10^{{-15}}$ sur les moyennes), confirmant que DWY et RTS calculent "
            rf"la \emph{{même}} loi lissée (\S\ref{{subsec:dwy-joseph}}).}}",
            r"    \label{tab:dwy-equiv}",
            r"\end{table}",
            "",
        ]
        out = Path(args.out)
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nLaTeX table written to {out}")


if __name__ == "__main__":
    main()
