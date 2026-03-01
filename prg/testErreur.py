#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    # pour EPKF, UPKF, PPF
    python3 prg/testErreur.py --N 100 --sigmaSet "wan2000" --nbParticles 500 --verbose 0 --nonLinearModelName "x1_y1_gordon"
    # pour PKF
    python3 prg/testErreur.py --N 100 --linearModelName "A_mQ_x1_y1" --verbose 0
"""

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from base_classes.nonlinear_epkf_runner_simulation import BaseNonLinearEPKFRunnerSim
from base_classes.nonlinear_upkf_runner_simulation import BaseNonLinearUPKFRunnerSim
from base_classes.nonlinear_ppf_runner_simulation import BaseNonLinearPPFRunnerSim
from base_classes.linear_pkf_runner_simulation import LinearPKFRunnerSim

from others.parser import addParseToParser
from others.plot_settings import DPI, FACECOLOR


# ==========================================================
# Types
# ==========================================================


@dataclass
class FilterStats:
    tr_mean: np.ndarray
    tr_std: np.ndarray
    err_mean: np.ndarray
    err_std: np.ndarray


# ==========================================================
# Main class
# ==========================================================


class FilterErrorAnalyzer:
    """
    Monte-Carlo analyser that compares the covariance trace of a Kalman-family
    filter against the actual estimation error over N time steps.

    Parameters
    ----------
    filter_name : str
        One of "EPKF", "UPKF", "PPF", "PKF".
    nb_exp : int
        Number of Monte-Carlo runs.
    save_path : str
        Where to write the output plot.
    """

    # --- class-level registries (shared by all instances) ---

    _RUNNERS: dict = {
        "EPKF": BaseNonLinearEPKFRunnerSim,
        "UPKF": BaseNonLinearUPKFRunnerSim,
        "PPF": BaseNonLinearPPFRunnerSim,
        "PKF": LinearPKFRunnerSim,
    }

    _EXTRA_KWARGS: dict = {
        "UPKF": lambda args: {"sigmaSet": args.sigmaSet},
        "PPF": lambda args: {"nbParticles": args.nbParticles},
    }

    # -------------------------------------------------------

    def __init__(
        self,
        filter_name: str,
        nb_exp: int = 200,
        save_dir: str = "./data/plot",
    ):
        if filter_name not in self._RUNNERS:
            raise ValueError(
                f"Unknown filter '{filter_name}'. "
                f"Valid options: {list(self._RUNNERS)}"
            )

        self.filter_name = filter_name
        self.nb_exp = nb_exp
        self.save_path = f"{save_dir}/{filter_name}_N{nb_exp}.png"

        # Populated after parse() / run()
        self.args = None
        self.model_name: str | None = None
        self.stats: FilterStats | None = None
        self.nb_valid: int = 0
        self.nb_failed: int = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def parse(self) -> "FilterErrorAnalyzer":
        """Parse CLI arguments and store them in self.args / self.model_name."""
        parser = argparse.ArgumentParser(description="Run Linear/NonLinear filters")

        addParseToParser(
            parser,
            [
                "nonLinearModelName",
                "linearModelName",
                "N",
                "sKey",
                "sigmaSet",
                "nbParticles",
                "dataFileName",
            ],
        )

        args = parser.parse_args()

        if args.dataFileName and args.N:
            parser.error("--N should not be used with --dataFileName")

        if bool(args.linearModelName) == bool(args.nonLinearModelName):
            parser.error("Use exactly one of --linearModelName or --nonLinearModelName")

        if args.N is None:
            print(
                "Error: --N is required when not using --dataFileName", file=sys.stderr
            )
            sys.exit(1)

        self.args = args
        self.model_name = args.nonLinearModelName or args.linearModelName
        return self  # allows chaining: analyzer.parse().run().plot()

    def run(self) -> "FilterErrorAnalyzer":
        """Run all Monte-Carlo experiments and compute statistics.

        If a run raises a RuntimeError, it is skipped and excluded from
        statistics. A warning is logged with the failure count at the end.
        """
        if self.args is None:
            raise RuntimeError("Call parse() before run().")

        N = self.args.N
        traces_valid = []
        errors_valid = []
        nb_failed = 0

        for i in tqdm(range(self.nb_exp), desc="Monte-Carlo runs", unit="run"):
            runner = self._build_runner()
            try:
                hist = runner.run(i)
            except RuntimeError as rte:
                if self.args.verbose > 0:
                    self.logger.error("Run %d failed and was skipped: %s", i, rte)
                nb_failed += 1
                continue

            for n in range(N + 1):
                # np.sum : erreur quadratique totale sur le vecteur d'état
                # (cohérent avec trace(P) = somme des variances par composante)
                err_n = np.sum((hist[n]["xkp1"] - hist[n]["Xkp1_update"]) ** 2)
                if err_n > 20:
                    print(
                        "n=",
                        n,
                        ", Erreur=",
                        err_n,
                        "Trace=",
                        np.trace(hist[n]["PXXkp1_update"]),
                    )
                    print()
                    input("ATTENTE")

            traces_valid.append(
                [np.trace(hist[n]["PXXkp1_update"]) for n in range(N + 1)]
            )
            errors_valid.append(
                [
                    # np.sum(diff²) : somme des erreurs quadratiques sur toutes
                    # les composantes du vecteur d'état → comparable à trace(P)
                    # (np.mean diviserait par dim, sous-estimant d'un facteur dim)
                    np.sum((hist[n]["xkp1"] - hist[n]["Xkp1_update"]) ** 2)
                    for n in range(N + 1)
                ]
            )

        nb_valid = len(traces_valid)

        if nb_valid == 0:
            raise RuntimeError(
                f"All {self.nb_exp} Monte-Carlo runs failed — cannot compute statistics."
            )

        if nb_failed > 0:
            self.logger.warning(
                "%d / %d runs failed and were excluded from statistics.",
                nb_failed,
                self.nb_exp,
            )

        traces_all = np.array(traces_valid)  # shape (nb_valid, N+1)
        errors_all = np.array(errors_valid)  # shape (nb_valid, N+1)

        self.stats = FilterStats(
            tr_mean=traces_all.mean(axis=0),
            tr_std=traces_all.std(axis=0),
            err_mean=errors_all.mean(axis=0),
            err_std=errors_all.std(axis=0),
        )
        self.nb_valid = nb_valid
        self.nb_failed = nb_failed

        return self

    def plot(self) -> "FilterErrorAnalyzer":
        """Save a plot of trace(Pxx) vs MSE with 95% confidence intervals."""
        if self.stats is None:
            raise RuntimeError("Call run() before plot().")

        plt.style.use("seaborn-v0_8")

        stats = self.stats
        n = np.arange(len(stats.tr_mean))
        coef = 1.96 / np.sqrt(self.nb_valid)

        fig, ax = plt.subplots(figsize=(6, 3), facecolor=FACECOLOR)

        # trace(Pxx)
        ax.plot(n, stats.tr_mean, linewidth=2.5, label=r"trace($P^{xx}_{n \mid n}$)")
        ax.fill_between(
            n,
            stats.tr_mean - coef * stats.tr_std,
            stats.tr_mean + coef * stats.tr_std,
            alpha=0.25,
        )

        # Erreur quadratique totale
        ax.plot(
            n,
            stats.err_mean,
            linestyle="--",
            linewidth=2.5,
            label=r"$\left\| x_n - \hat{x}_n \right\|^2$",
        )
        ax.fill_between(
            n,
            stats.err_mean - coef * stats.err_std,
            stats.err_mean + coef * stats.err_std,
            alpha=0.25,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        title = (
            f"[{self.filter_name}] Covariance Trace vs Estimation Error (95% CI)\n"
            f"Runs : {self.nb_valid} / {self.nb_exp}"
        )
        if self.nb_failed > 0:
            title += f"  —  ({self.nb_failed} failed)"
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.save_path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close("all")

        print(f"Plot saved to {self.save_path}")
        return self

    # -------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------

    def _build_runner(self):
        """Instantiate the appropriate runner with the correct kwargs."""
        runner_class = self._RUNNERS[self.filter_name]

        base_kwargs = dict(
            model_name=self.model_name,
            N=self.args.N,
            sKey=None,
            verbose=self.args.verbose,
            plot=False,
            save_history=True,
        )
        extra_kwargs = self._EXTRA_KWARGS.get(self.filter_name, lambda _: {})(self.args)

        return runner_class(**base_kwargs, **extra_kwargs)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    (
        FilterErrorAnalyzer(filter_name="UPKF", nb_exp=200, save_dir="./data/plot")
        .parse()
        .run()
        .plot()
    )
