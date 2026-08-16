"""Timing/memory benchmark of the six pairwise smoothers behind Linear_PKS(method=...):
confirms the O(N) cost the paper announces but never quantifies, and reports wall-clock +
peak memory. Drives the ACTUAL library on the shipped x1y1 pairwise model. Needs
AWESOMEPKF_ROOT (or run from a tree where prg is importable). Output: figures/bench_smoothers.pdf"""
import time
import tracemalloc
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "lines.linewidth": 1.3, "lines.markersize": 3.5})
import matplotlib.pyplot as plt

from prg.classes.linear_pkf import Linear_PKF
from prg.classes.linear_pks import Linear_PKS
from prg.classes.param_linear import ParamLinear
from prg.models.linear import ModelFactoryLinear

OUT = str(Path(__file__).resolve().parents[1] / "figures"); Path(OUT).mkdir(parents=True, exist_ok=True)
PD_MQ = np.array([[0.10, 0.02], [0.02, 0.08]])
METHODS = ["RTS", "BF", "MBF", "2F", "DWY", "VAR"]


def make_param():
    p = ModelFactoryLinear.create("model_x1_y1_AQ_pairwise").get_params().copy()
    dx, dy = p.pop("dim_x"), p.pop("dim_y")
    p["B"] = np.eye(dx + dy); p["mQ"] = PD_MQ.copy()
    return ParamLinear(0, dx, dy, **p)


def time_method(param, sim, method, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        Linear_PKS(param, method=method).process_N_data_smoother(N=None, data_generator=iter(sim))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def peak_mem(param, sim, method):
    tracemalloc.start()
    Linear_PKS(param, method=method).process_N_data_smoother(N=None, data_generator=iter(sim))
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return peak / 1e6   # MB


def main():
    param = make_param()
    Ns = [500, 1000, 2000, 4000, 8000, 16000]
    times = {m: [] for m in METHODS}
    for N in Ns:
        sim = list(Linear_PKF(param, sKey=1).simulate_N_data(N))
        time_method(param, sim[:min(len(sim), 500)], "RTS", reps=1)   # warm up
        for m in METHODS:
            times[m].append(time_method(param, sim, m))
        print(f"N={N:6d}: " + "  ".join(f"{m}={times[m][-1]*1e3:6.1f}ms" for m in METHODS), flush=True)

    # peak memory at the largest N
    simL = list(Linear_PKF(param, sKey=1).simulate_N_data(Ns[-1]))
    mem = {m: peak_mem(param, simL, m) for m in METHODS}
    print(f"peak memory (MB) at N={Ns[-1]}: " + "  ".join(f"{m}={mem[m]:.1f}" for m in METHODS))
    # empirical scaling exponent (log-log slope) per method
    lnN = np.log(Ns)
    for m in METHODS:
        slope = np.polyfit(lnN, np.log(times[m]), 1)[0]
        print(f"  {m}: O(N^{slope:.2f})")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    # absolute Okabe-Ito colours (the awesomepkf prop_cycle would otherwise map C0->red, repeating every 3)
    COL = {"RTS": "#0072B2", "BF": "#E69F00", "MBF": "#009E73",
           "2F": "#D55E00", "DWY": "#CC79A7", "VAR": "#56B4E9"}
    styles = {"RTS": "o", "BF": "s", "MBF": "^", "2F": "v", "DWY": "D", "VAR": "P"}
    for m in METHODS:
        ax[0].loglog(Ns, np.array(times[m]) * 1e3, "-", marker=styles[m], color=COL[m], label=m)
    ref = np.array(Ns, float); ref = ref / ref[0] * times["MBF"][0] * 1e3
    ax[0].loglog(Ns, ref, "k--", lw=1, alpha=0.6, label=r"$O(N)$ guide")
    ax[0].set_xlabel(r"record length $N$"); ax[0].set_ylabel("smoothing time (ms)")
    ax[0].set_title("(a) all six smoothers are $O(N)$")
    ax[0].legend(ncol=2, loc="upper left"); ax[0].grid(alpha=0.3, which="both")

    yv = np.arange(len(METHODS))[::-1]
    ax[1].barh(yv, [mem[m] for m in METHODS], color=[COL[m] for m in METHODS], alpha=0.9, height=0.6)
    ax[1].set_yticks(yv); ax[1].set_yticklabels(METHODS)
    ax[1].set_xlabel(f"peak memory (MB), $N={Ns[-1]}$")
    ax[1].set_title("(b) peak memory")
    ax[1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(OUT + "/bench_smoothers.pdf")
    fig.savefig(OUT + "/bench_smoothers_preview.png", dpi=150)
    print("saved", OUT + "/bench_smoothers.pdf")


if __name__ == "__main__":
    main()
