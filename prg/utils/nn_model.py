"""
Neural-network based nonlinear model.

Learns the dynamics  z_{k+1} = g_nn(z_k)  from a CSV of noisy 2-D
observations, then exposes the standard PKF interface (g, jacobiens_g)
via BaseModelNonLinear.

Noise model is additive:  g(z, noise, dt) = g_nn(z) + B · noise,  B = I.
"""

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from prg.models.nonLinear.base_model_nonLinear import BaseModelNonLinear
from prg.utils.exceptions import NumericalError

__all__ = ["NNModel"]


# ======================================================================
# MLP definition (only defined when torch is available)
# ======================================================================
if _TORCH_AVAILABLE:
    class _MLP(nn.Module):
        """Simple fully-connected network  R^dim_in → R^dim_out."""

        def __init__(self, dim_in, dim_out, hidden_sizes=(64, 64), activation=nn.Tanh):
            super().__init__()
            layers = []
            prev = dim_in
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(activation())
                prev = h
            layers.append(nn.Linear(prev, dim_out))
            self.net = nn.Sequential(*layers)

        def forward(self, z):
            return self.net(z)


# ======================================================================
# NNModel — PKF-compatible wrapper
# ======================================================================
class NNModel(BaseModelNonLinear):
    """
    Non-linear model whose dynamics are learned by a neural network.

    Usage
    -----
    >>> model = NNModel(csv_path="data/datafile/testNL.csv",
    ...                 dim_x=1, dim_y=1,
    ...                 hidden_sizes=(64, 64), epochs=500, lr=1e-3)
    >>> z      = np.array([[0.5], [0.3]])        # (2, 1)
    >>> noise  = np.zeros((2, 1))
    >>> z_next = model.g(z, noise, dt=1.0)       # (2, 1)
    >>> An, Bn = model.jacobiens_g(z, noise, 1.0)

    Parameters
    ----------
    csv_path      : path to a CSV with columns X0, Y0 (consecutive time steps).
    dim_x, dim_y  : state / observation dimensions  (default 1, 1).
    hidden_sizes  : tuple of hidden-layer widths for the MLP.
    epochs        : number of training epochs.
    lr            : learning rate (Adam).
    batch_size    : mini-batch size (0 = full batch).
    verbose       : print training progress every *verbose* epochs (0 = silent).
    seed          : random seed for reproducibility.
    """

    MODEL_NAME = "nNModel"

    def __init__(
        self,
        csv_path,
        dim_x=1,
        dim_y=1,
        hidden_sizes=(64, 64),
        epochs=500,
        lr=1e-3,
        batch_size=0,
        verbose=50,
        seed=42,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "NNModel requires PyTorch. Install it with:\n"
                "  pip install awesomepkf[nn]\n"
                "or:\n"
                "  pip install torch"
            )
        super().__init__(dim_x, dim_y, model_type="nonlinear")
        self.pairwiseModel = True

        torch.manual_seed(seed)
        # Seed numpy's global RNG so any np.random.* call elsewhere in the
        # NN training pipeline (e.g. inside torch internals or downstream
        # transforms) is reproducible. Generator-based local RNG would not
        # propagate to those callers.
        np.random.seed(seed)  # noqa: NPY002 — reproducibility of global RNG is intentional

        # ── Load data & build training pairs ──────────────────────────
        data = self._load_csv(csv_path)  # (T, dim_xy)
        Z_in = data[:-1]                 # z_k      (T-1, dim_xy)
        Z_out = data[1:]                 # z_{k+1}  (T-1, dim_xy)

        self._z_mean = Z_in.mean(axis=0)
        self._z_std = Z_in.std(axis=0) + 1e-8

        Z_in_n = (Z_in - self._z_mean) / self._z_std
        Z_out_n = (Z_out - self._z_mean) / self._z_std

        X_train = torch.tensor(Z_in_n, dtype=torch.float32)
        Y_train = torch.tensor(Z_out_n, dtype=torch.float32)

        # ── Build & train the network ─────────────────────────────────
        self._net = _MLP(self.dim_xy, self.dim_xy, hidden_sizes)
        self._train(X_train, Y_train, epochs, lr, batch_size, verbose)

        # ── Estimate mQ from residuals ────────────────────────────────
        self._net.eval()
        with torch.no_grad():
            pred_n = self._net(X_train).numpy()
        residuals_n = Y_train.numpy() - pred_n
        residuals = residuals_n * self._z_std
        self.mQ = np.cov(residuals, rowvar=False)
        if self.mQ.ndim == 0:
            self.mQ = self.mQ.reshape(1, 1)

        # ── Initial conditions ────────────────────────────────────────
        self.mz0 = data[0].reshape(self.dim_xy, 1)
        self.Pz0 = self.mQ.copy()

        # ── Noise Jacobian is identity (additive noise) ───────────────
        self._Bn = np.eye(self.dim_xy)

        # ── Precompute torch tensors for mean/std ─────────────────────
        self._z_mean_t = torch.tensor(self._z_mean, dtype=torch.float32)
        self._z_std_t = torch.tensor(self._z_std, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_csv(csv_path):
        """
        Loads a CSV (header row expected) and returns an (T, dim) ndarray.

        Resolution order for the path:
          1. Exact path as given
          2. data/datafile/<basename>   (project-relative fallback)
        """
        candidates = [
            Path(csv_path),
            Path("data/datafile") / Path(csv_path).name,
        ]
        for path in candidates:
            if path.is_file():
                break
        else:
            tried = "\n  ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                f"NNModel: fichier introuvable. Chemins testés :\n  {tried}"
            )
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train(self, X, Y, epochs, lr, batch_size, verbose):
        optimiser = torch.optim.Adam(self._net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        N = X.shape[0]
        bs = N if batch_size <= 0 else batch_size

        self._net.train()
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, bs):
                idx = perm[i : i + bs]
                xb, yb = X[idx], Y[idx]
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
                n_batches += 1
            if verbose and epoch % verbose == 0:
                print(f"  [NNModel] epoch {epoch:4d}/{epochs}  "
                      f"loss = {epoch_loss / n_batches:.6e}")

        self._net.eval()

    # ------------------------------------------------------------------
    # Forward pass  (numpy ↔ torch bridge)
    # ------------------------------------------------------------------
    def _forward_np(self, z_np):
        """
        Evaluates the learned dynamics on numpy input(s).

        Parameters
        ----------
        z_np : (dim_xy,) or (N, dim_xy)

        Returns
        -------
        out : same shape as z_np
        """
        squeeze = z_np.ndim == 1
        if squeeze:
            z_np = z_np[np.newaxis, :]
        z_t = torch.tensor(z_np, dtype=torch.float32)
        z_n = (z_t - self._z_mean_t) / self._z_std_t
        with torch.no_grad():
            out_n = self._net(z_n)
        out = out_n * self._z_std_t + self._z_mean_t
        out_np = out.numpy()
        return out_np.squeeze(0) if squeeze else out_np

    # ------------------------------------------------------------------
    # Jacobian via torch.autograd
    # ------------------------------------------------------------------
    def _jacobian_np(self, z_np):
        """
        Computes  dg_nn/dz  at a single point z  via torch autograd.

        Parameters
        ----------
        z_np : (dim_xy,)

        Returns
        -------
        J : (dim_xy, dim_xy)
        """
        z_t = torch.tensor(z_np, dtype=torch.float32).requires_grad_(True)
        z_n = (z_t - self._z_mean_t) / self._z_std_t
        out_n = self._net(z_n)
        # out = out_n * std + mean  →  d(out)/d(z) = std_out/std_in * d(out_n)/d(z_n)
        # but std_out == std_in (same normalisation), so the chain rule gives:
        # d(out)/d(z) = (std/std) * d(out_n)/d(z_n)  — but we compute directly:
        out = out_n * self._z_std_t + self._z_mean_t

        nz = self.dim_xy
        J = torch.zeros(nz, nz)
        for i in range(nz):
            if z_t.grad is not None:
                z_t.grad.zero_()
            out[i].backward(retain_graph=True)
            J[i] = z_t.grad.clone()
        return J.numpy()

    # ------------------------------------------------------------------
    # PKF interface:  _g  and  _jacobiens_g
    # ------------------------------------------------------------------
    def _g(self, x, y, nx, ny, dt):
        """
        g(z, noise) = g_nn(z) + noise   (additive noise).

        2D : x (dim_x,1), y (dim_y,1), … → (dim_xy, 1)
        3D : x (N,dim_x,1), …            → (N, dim_xy, 1)
        """
        try:
            if x.ndim == 2:
                z = np.vstack((x, y))[:, 0]            # (dim_xy,)
                noise = np.vstack((nx, ny))             # (dim_xy, 1)
                out = self._forward_np(z).reshape(self.dim_xy, 1)
                return out + noise
            N = x.shape[0]
            z = np.concatenate((x, y), axis=1)[:, :, 0]   # (N, dim_xy)
            noise = np.concatenate((nx, ny), axis=1)       # (N, dim_xy, 1)
            out = self._forward_np(z).reshape(N, self.dim_xy, 1)
            return out + noise
        except Exception as e:
            raise NumericalError(
                f"[NNModel] _g: {type(e).__name__}: {e}"
            ) from e

    def _jacobiens_g(self, x, y, nx, ny, dt):
        """
        Jacobians of g(z, noise) = g_nn(z) + noise.

            An = dg/dz     = dg_nn/dz   (state Jacobian)
            Bn = dg/dnoise = I           (additive noise)

        2D : → (dim_xy, dim_xy), (dim_xy, dim_xy)
        3D : → (N, dim_xy, dim_xy), (N, dim_xy, dim_xy)
        """
        try:
            if x.ndim == 2:
                z = np.vstack((x, y))[:, 0]             # (dim_xy,)
                An = self._jacobian_np(z)                # (dim_xy, dim_xy)
                return An, self._Bn
            N = x.shape[0]
            z = np.concatenate((x, y), axis=1)[:, :, 0]  # (N, dim_xy)
            An = np.empty((N, self.dim_xy, self.dim_xy))
            for i in range(N):
                An[i] = self._jacobian_np(z[i])
            Bn = np.tile(self._Bn, (N, 1, 1))
            return An, Bn
        except Exception as e:
            raise NumericalError(
                f"[NNModel] _jacobiens_g: {type(e).__name__}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # LaTeX (minimal)
    # ------------------------------------------------------------------
    def latex_model(self) -> str:
        nx, ny = self.dim_x, self.dim_y
        x_n = r"\mathbf{x}" if nx > 1 else "x"
        y_n = r"\mathbf{y}" if ny > 1 else "y"
        z_n = r"\mathbf{z}"
        v_n = r"\mathbf{v}"
        return (
            r"\begin{aligned}"
            rf"  {z_n}_{{k+1}} &= g_\mathrm{{NN}}\!\left({z_n}_k\right)"
            rf" + {v_n}_k, \qquad"
            rf" {v_n}_k \sim \mathcal{{N}}(0,\,\hat{{\mathcal{{Q}}}}) \\"
            rf"  {z_n} &= ({x_n},\,{y_n})"
            r"\end{aligned}"
        )


# ======================================================================
# Programme principal de test
# ======================================================================
if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    from prg.utils.plot_settings import BIG_SIZE, DPI, FACECOLOR

    parser = argparse.ArgumentParser(description="Test NNModel on a 2-D CSV file.")
    parser.add_argument(
        "csv", nargs="?",
        default="data/datafile/testNL.csv",
        help="Path to the CSV file (header X0,Y0). Default: data/datafile/testNL.csv",
    )
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--hidden",       type=int,   nargs="+", default=[64, 64])
    parser.add_argument("--n-grid",       type=int,   default=60,
                        help="Resolution of the evaluation grid")
    parser.add_argument("--out",          type=str,   default="data/plot",
                        help="Output directory for the figure")
    args = parser.parse_args()

    # ── 1. Entraînement ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fichier  : {args.csv}")
    print(f"  Couches  : {args.hidden}   epochs={args.epochs}   lr={args.lr}")
    print(f"{'='*60}\n")

    model = NNModel(
        csv_path=args.csv,
        dim_x=1, dim_y=1,
        hidden_sizes=tuple(args.hidden),
        epochs=args.epochs,
        lr=args.lr,
        verbose=max(1, args.epochs // 5),
    )

    print(f"\nmQ estimée :\n{model.mQ}")
    print(f"mz0 = {model.mz0.ravel()}")

    # ── 2. Données brutes ─────────────────────────────────────────────
    data = NNModel._load_csv(args.csv)          # (T, 2)
    X_data, Y_data = data[:, 0], data[:, 1]

    # Plage d'évaluation : légèrement élargie autour des données
    x_min, x_max = X_data.min(), X_data.max()
    y_min, y_max = Y_data.min(), Y_data.max()
    margin_x = 0.1 * (x_max - x_min)
    margin_y = 0.1 * (y_max - y_min)
    xg = np.linspace(x_min - margin_x, x_max + margin_x, args.n_grid)
    yg = np.linspace(y_min - margin_y, y_max + margin_y, args.n_grid)
    XG, YG = np.meshgrid(xg, yg)                 # (n, n)
    Z_grid = np.stack([XG.ravel(), YG.ravel()], axis=1)   # (n², 2)

    # ── 3. Évaluation de g_nn et du Jacobien sur la grille ───────────
    G_grid = model._forward_np(Z_grid)            # (n², 2)
    GX = G_grid[:, 0].reshape(args.n_grid, args.n_grid)   # g_x(x,y)
    GY = G_grid[:, 1].reshape(args.n_grid, args.n_grid)   # g_y(x,y)

    # Jacobien : An[i] = [[∂gx/∂x, ∂gx/∂y], [∂gy/∂x, ∂gy/∂y]]
    print("\nCalcul du Jacobien sur la grille…")
    n2 = Z_grid.shape[0]
    AN = np.empty((n2, 2, 2))
    for i in range(n2):
        AN[i] = model._jacobian_np(Z_grid[i])
    dgx_dx = AN[:, 0, 0].reshape(args.n_grid, args.n_grid)
    dgx_dy = AN[:, 0, 1].reshape(args.n_grid, args.n_grid)
    dgy_dx = AN[:, 1, 0].reshape(args.n_grid, args.n_grid)
    dgy_dy = AN[:, 1, 1].reshape(args.n_grid, args.n_grid)
    print("Jacobien calculé.")

    # ── 4. Figure ─────────────────────────────────────────────────────
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10), facecolor=FACECOLOR)
    fig.suptitle("NNModel — fonction apprise et jacobienne", fontsize=BIG_SIZE + 2,
                 fontweight="bold")

    def _add_data_scatter(ax, alpha=0.15):
        """Superpose les données brutes (x_k, y_k) sur un axe 2D."""
        ax.scatter(X_data, Y_data, s=4, c="white", alpha=alpha, zorder=5,
                   label="données")

    def _cbar(fig, ax, im, title):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, size=BIG_SIZE)
        ax.set_xlabel(r"$x_k$")
        ax.set_ylabel(r"$y_k$")

    # ── Ligne 1 : surfaces 3D de g_x et g_y
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    ax1.plot_surface(XG, YG, GX, cmap="viridis", alpha=0.85)
    ax1.set_title(r"$g_x(x,y)$", size=BIG_SIZE)
    ax1.set_xlabel(r"$x_k$"); ax1.set_ylabel(r"$y_k$"); ax1.set_zlabel(r"$x_{k+1}$")
    ax1.view_init(30, -60)

    ax2 = fig.add_subplot(2, 4, 2, projection="3d")
    ax2.plot_surface(XG, YG, GY, cmap="plasma", alpha=0.85)
    ax2.set_title(r"$g_y(x,y)$", size=BIG_SIZE)
    ax2.set_xlabel(r"$x_k$"); ax2.set_ylabel(r"$y_k$"); ax2.set_zlabel(r"$y_{k+1}$")
    ax2.view_init(30, -60)

    # ── Cartes 2D de g_x et g_y avec scatter des données
    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.contourf(XG, YG, GX, levels=25, cmap="viridis")
    _add_data_scatter(ax3)
    _cbar(fig, ax3, im3, r"$g_x(x,y)$")

    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.contourf(XG, YG, GY, levels=25, cmap="plasma")
    _add_data_scatter(ax4)
    _cbar(fig, ax4, im4, r"$g_y(x,y)$")

    # ── Ligne 2 : 4 entrées du Jacobien An
    panels = [
        (fig.add_subplot(2, 4, 5), dgx_dx, r"$\partial g_x/\partial x$", "RdBu_r"),
        (fig.add_subplot(2, 4, 6), dgx_dy, r"$\partial g_x/\partial y$", "RdBu_r"),
        (fig.add_subplot(2, 4, 7), dgy_dx, r"$\partial g_y/\partial x$", "RdBu_r"),
        (fig.add_subplot(2, 4, 8), dgy_dy, r"$\partial g_y/\partial y$", "RdBu_r"),
    ]
    for ax, data_p, title, cmap in panels:
        vmax = np.abs(data_p).max()
        im = ax.contourf(XG, YG, data_p, levels=25, cmap=cmap,
                         vmin=-vmax, vmax=vmax)
        _add_data_scatter(ax, alpha=0.10)
        _cbar(fig, ax, im, title)

    plt.tight_layout()
    out_path = str(Path(args.out) / "nn_model_test.png")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
    plt.close(fig)
    print(f"\nFigure sauvegardée → {out_path}")
