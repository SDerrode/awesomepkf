import logging

import numpy as np

from prg.utils.numerics import EIG_TOL_FAIL, EIG_TOL_WARN

logger = logging.getLogger(__name__)


def check_eigvals(eigvals: np.ndarray) -> None:
    """
    Validate eigenvalues against positivity tolerances.

    Raises an error if any eigenvalue falls below ``EIG_TOL_FAIL``.
    Eigenvalues between ``EIG_TOL_FAIL`` and ``EIG_TOL_WARN`` are considered
    numerical noise and are silently accepted.

    Parameters
    ----------
    eigvals : np.ndarray
        Sorted array of eigenvalues, shape ``(n,)``.

    Raises
    ------
    ValueError
        If any eigenvalue is below ``EIG_TOL_FAIL``.
    """
    if np.any(eigvals < EIG_TOL_FAIL):
        raise ValueError(
            f"Matrix is not positive semi-definite: "
            f"negative eigenvalues = {eigvals[eigvals < EIG_TOL_FAIL]}"
        )
    if np.any(eigvals < EIG_TOL_WARN):
        logger.warning(
            "Near-zero eigenvalues detected (below EIG_TOL_WARN): - %s — likely numerical noise.",
            eigvals[eigvals < EIG_TOL_WARN],
        )


def generate_block_matrix(
    rng: np.random.Generator,
    dim_x: int,
    dim_y: int,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Randomly generates a block matrix M of dimension (dim_x + dim_y) x (dim_x + dim_y):

        M = | A   B  |
            | B^T D  |

    with the following constraints:
        - A  : covariance matrix (symmetric positive semi-definite)
        - D  : invertible matrix (here positive definite to guarantee M >= 0)
        - S  : Schur complement S = A - B D^{-1} B^T is a covariance matrix
        - M  : strictly diagonally dominant
        - All values of M are in [-threshold, threshold]

    Parameters
    ----------
    rng       : np.random.Generator  — random number generator (e.g. np.random.default_rng(42))
    dim_x     : int                  — dimension of block A (and B rows)
    dim_y     : int                  — dimension of block D (and B columns)
    threshold : float                — absolute bound on all values of M (default: 1.0)
                                       Example: threshold=0.1  =>  all values in [-0.1, 0.1]

    Returns
    -------
    M : np.ndarray of shape (dim_x + dim_y, dim_x + dim_y)
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be strictly positive, got: {threshold}")
    n = dim_x + dim_y
    # Scaling factor: the entire internal construction is proportional to the threshold
    t = threshold

    # ------------------------------------------------------------------
    # 1. Construction of D: symmetric positive definite
    # ------------------------------------------------------------------
    raw_D = rng.uniform(-0.3 * t, 0.3 * t, size=(dim_y, dim_y))
    D = raw_D.T @ raw_D
    D = _make_diag_dominant_and_bounded(D, rng, dim_y, t)

    # ------------------------------------------------------------------
    # 2. Construction of B
    # ------------------------------------------------------------------
    D_inv = np.linalg.inv(D)
    b_scale = 0.15 * t
    B = rng.uniform(-b_scale, b_scale, size=(dim_x, dim_y))

    # ------------------------------------------------------------------
    # 3. Construction of A via the Schur complement
    # ------------------------------------------------------------------
    raw_S = rng.uniform(-0.2 * t, 0.2 * t, size=(dim_x, dim_x))
    S = raw_S.T @ raw_S
    A = S + B @ D_inv @ B.T
    A = (A + A.T) / 2
    A = _make_diag_dominant_and_bounded(A, rng, dim_x, t)

    # ------------------------------------------------------------------
    # 4. Assembly of M
    # ------------------------------------------------------------------
    M = np.zeros((n, n))
    M[:dim_x, :dim_x] = A
    M[:dim_x, dim_x:] = B
    M[dim_x:, :dim_x] = B.T
    M[dim_x:, dim_x:] = D

    # ------------------------------------------------------------------
    # 5. Passe finale : diagonale dominante + borne threshold
    # ------------------------------------------------------------------
    M = _enforce_diag_dominant_and_bounded(M, t)

    # ------------------------------------------------------------------
    # 6. Verification
    # ------------------------------------------------------------------
    _verify(M, dim_x, dim_y, t)

    return M


# ── Utility functions ─────────────────────────────────────────────────────────


def _make_diag_dominant_and_bounded(
    X: np.ndarray, rng: np.random.Generator, dim: int, threshold: float = 1.0
) -> np.ndarray:
    """Makes X symmetric, diagonally dominant and bounded in [-threshold, threshold]."""
    X = (X + X.T) / 2
    # Clip off-diagonal to 40% of threshold to leave margin for the diagonal
    off_clip = 0.4 * threshold
    np.fill_diagonal(X, 0)
    X = np.clip(X, -off_clip, off_clip)
    # Diagonal = sum of absolute values of its row + small epsilon
    row_sums = np.abs(X).sum(axis=1)
    epsilon = rng.uniform(0.02 * threshold, 0.1 * threshold, size=dim)
    np.fill_diagonal(X, row_sums + epsilon)
    # Normalise to stay within [-threshold, threshold]
    max_val = np.abs(X).max()
    if max_val > threshold:
        X = X / max_val * threshold * 0.99
    return X


def _enforce_diag_dominant_and_bounded(
    M: np.ndarray, threshold: float = 1.0
) -> np.ndarray:
    """
    Final pass: guarantees that all values are in [-threshold, threshold]
    and that M is diagonally dominant (row-wise).
    """
    n = M.shape[0]
    M = np.clip(M, -threshold, threshold)
    for i in range(n):
        off_diag_sum = np.abs(M[i]).sum() - abs(M[i, i])
        if M[i, i] <= off_diag_sum:
            M[i, i] = off_diag_sum + 1e-6
        if M[i, i] > threshold:
            scale = (threshold - 1e-6) / off_diag_sum if off_diag_sum > 0 else 1.0
            for j in range(n):
                if j != i:
                    M[i, j] *= scale
                    M[j, i] = M[i, j]
            M[i, i] = np.abs(M[i]).sum() - abs(M[i, i]) + 1e-6
            M[i, i] = min(M[i, i], threshold - 1e-9)
    return M


def _verify(M: np.ndarray, dim_x: int, dim_y: int, threshold: float = 1.0):
    """Verifies the properties of the generated matrix."""
    n = dim_x + dim_y
    A = M[:dim_x, :dim_x]
    B = M[:dim_x, dim_x:]
    D = M[dim_x:, dim_x:]

    # Symmetry of M
    assert np.allclose(M, M.T, atol=1e-9), "M is not symmetric"

    # Bounded values
    assert (
        np.abs(M).max() <= threshold + 1e-9
    ), f"Value out of bounds: {np.abs(M).max():.6f} > {threshold}"

    # Diagonally dominant (row-wise)
    for i in range(n):
        off = np.abs(M[i]).sum() - abs(M[i, i])
        assert M[i, i] > off - 1e-9, f"Diagonal not dominant at row {i}"

    # D invertible
    assert abs(np.linalg.det(D)) > 1e-10, "D is not invertible"

    # A SDP
    eigs_A = np.linalg.eigvalsh(A)
    check_eigvals(eigs_A)
    assert eigs_A.min() >= -1e-9, f"A not SDP, min eigenvalue = {eigs_A.min():.2e}"

    # SDP Schur complement
    S = A - B @ np.linalg.inv(D) @ B.T
    eigs_S = np.linalg.eigvalsh(S)
    check_eigvals(eigs_S)
    assert (
        eigs_S.min() >= -1e-9
    ), f"Schur complement not SDP, min eigenvalue = {eigs_S.min():.2e}"

    # print("✓ All checks passed.")
    # print(f"  • max|M|      = {np.abs(M).max():.6f}  (≤ {threshold})")
    # print(f"  • min eig(A)  = {eigs_A.min():.6f}  (≥ 0, covariance)")
    # print(f"  • min eig(S)  = {eigs_S.min():.6f}  (≥ 0, Schur covariance)")
    # print(f"  • det(D)      = {np.linalg.det(D):.6f}  (≠ 0, invertible)")


# ── Usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    dim_x, dim_y = 3, 2

    for threshold in [1.0, 0.5, 0.1]:
        print(f"\n{'='*55}")
        print(f"  Exemple avec threshold = {threshold}")
        print(f"{'='*55}")
        M = generate_block_matrix(rng, dim_x, dim_y, threshold=threshold)
        print(f"\nMatrice M ({dim_x + dim_y}x{dim_x + dim_y}) :\n")
        print(np.array2string(M, precision=5, suppress_small=True))

        S = (
            M[:dim_x, :dim_x]
            - M[:dim_x, dim_x:] @ np.linalg.inv(M[dim_x:, dim_x:]) @ M[:dim_x, dim_x:].T
        )
        print(
            f"\nComplément de Schur S :\n{np.array2string(S, precision=5, suppress_small=True)}"
        )
