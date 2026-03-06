# REQ-BEMD-004: Surface fitting from scattered data (GPU variant)
# Python implementation of gridfit using CuPy (GPU-accelerated)
# Original MATLAB gridfit by John D'Errico
#
# This variant offloads the normal equations solve to the GPU using
# CuPy's sparse CG solver. Intended for comparison benchmarks.

import numpy as np
from scipy import sparse as sp_sparse

import cupy as cp
import cupyx.scipy.sparse as cp_sparse
from cupyx.scipy.sparse.linalg import cg as cp_cg, LinearOperator as CpLinearOperator


def _build_system_cpu(x, y, z, xnodes, ynodes, smoothness):
    """Build the interpolation + regularization system on CPU (reuses gridfit logic)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    xnodes = np.asarray(xnodes, dtype=float).copy().ravel()
    ynodes = np.asarray(ynodes, dtype=float).copy().ravel()

    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    n = len(x)
    if n < 3:
        raise ValueError("Insufficient data for surface estimation.")

    xnodes[0] = min(xnodes[0], x.min())
    xnodes[-1] = max(xnodes[-1], x.max())
    ynodes[0] = min(ynodes[0], y.min())
    ynodes[-1] = max(ynodes[-1], y.max())

    dx = np.diff(xnodes)
    dy = np.diff(ynodes)
    nx = len(xnodes)
    ny = len(ynodes)
    ngrid = nx * ny

    xscale = np.mean(dx)
    yscale = np.mean(dy)

    indx = np.searchsorted(xnodes, x, side='right') - 1
    indy = np.searchsorted(ynodes, y, side='right') - 1
    indx = np.clip(indx, 0, nx - 2)
    indy = np.clip(indy, 0, ny - 2)

    ind = indy + ny * indx

    tx = np.clip((x - xnodes[indx]) / dx[indx], 0, 1)
    ty = np.clip((y - ynodes[indy]) / dy[indy], 0, 1)

    k = tx > ty
    L = np.ones(n, dtype=int)
    L[k] = ny

    t1 = np.minimum(tx, ty)
    t2 = np.maximum(tx, ty)

    row_idx = np.tile(np.arange(n), 3)
    col_idx = np.concatenate([ind, ind + ny + 1, ind + L])
    values = np.concatenate([1 - t2, t1, t2 - t1])

    A = sp_sparse.csr_matrix((values, (row_idx, col_idx)), shape=(n, ngrid))
    rhs = z.copy()

    reg_parts = []

    if ny > 2:
        i_y, j_y = np.meshgrid(np.arange(nx), np.arange(1, ny - 1), indexing='ij')
        i_y = i_y.ravel()
        j_y = j_y.ravel()
        ind_y = j_y + ny * i_y
        dy1 = dy[j_y - 1] / yscale
        dy2 = dy[j_y] / yscale
        m_y = len(ind_y)

        row_y = np.tile(np.arange(m_y), 3)
        col_y = np.concatenate([ind_y - 1, ind_y, ind_y + 1])
        val_y = np.concatenate([
            -2.0 / (dy1 * (dy1 + dy2)),
            2.0 / (dy1 * dy2),
            -2.0 / (dy2 * (dy1 + dy2))
        ])
        reg_parts.append(sp_sparse.csr_matrix((val_y, (row_y, col_y)), shape=(m_y, ngrid)))

    if nx > 2:
        i_x, j_x = np.meshgrid(np.arange(1, nx - 1), np.arange(ny), indexing='ij')
        i_x = i_x.ravel()
        j_x = j_x.ravel()
        ind_x = j_x + ny * i_x
        dx1 = dx[i_x - 1] / xscale
        dx2 = dx[i_x] / xscale
        m_x = len(ind_x)

        row_x = np.tile(np.arange(m_x), 3)
        col_x = np.concatenate([ind_x - ny, ind_x, ind_x + ny])
        val_x = np.concatenate([
            -2.0 / (dx1 * (dx1 + dx2)),
            2.0 / (dx1 * dx2),
            -2.0 / (dx2 * (dx1 + dx2))
        ])
        reg_parts.append(sp_sparse.csr_matrix((val_x, (row_x, col_x)), shape=(m_x, ngrid)))

    if reg_parts:
        Areg = sp_sparse.vstack(reg_parts)
        nreg = Areg.shape[0]

        NA = sp_sparse.linalg.norm(A, ord=1)
        NR = sp_sparse.linalg.norm(Areg, ord=1)
        if NR > 0:
            scale = smoothness * NA / NR
        else:
            scale = smoothness

        A_full = sp_sparse.vstack([A, Areg * scale])
        rhs_full = np.concatenate([rhs, np.zeros(nreg)])
    else:
        A_full = A
        rhs_full = rhs

    A_full_csc = A_full.tocsc()
    AtA = (A_full_csc.T @ A_full_csc + 1e-10 * sp_sparse.eye(ngrid, format='csc')).tocsc()
    Atrhs = A_full_csc.T @ rhs_full

    return AtA, Atrhs, nx, ny


def gridfit_gpu(x, y, z, xnodes, ynodes, smoothness=1.0, tol=1e-8,
                maxiter=None):
    """Estimates a surface on a 2D grid from scattered data using GPU CG solver.

    Builds the system on CPU, transfers to GPU, solves with CuPy's CG solver.

    Args:
        x: 1D array of x coordinates of scattered data.
        y: 1D array of y coordinates of scattered data.
        z: 1D array of z values at (x, y) points.
        xnodes: 1D array defining grid nodes in x direction.
        ynodes: 1D array defining grid nodes in y direction.
        smoothness: Smoothing parameter (default 1.0). Larger = smoother.
        tol: Convergence tolerance for CG solver (default 1e-8).
        maxiter: Maximum CG iterations (default: ngrid).

    Returns:
        zgrid: 2D numpy array (len(ynodes) x len(xnodes)) containing the fitted surface.
    """
    # Build system on CPU
    AtA, Atrhs, nx, ny = _build_system_cpu(x, y, z, xnodes, ynodes, smoothness)
    ngrid = nx * ny

    if maxiter is None:
        maxiter = ngrid

    # Transfer to GPU
    AtA_gpu = cp_sparse.csc_matrix(AtA)
    Atrhs_gpu = cp.asarray(Atrhs)

    # Jacobi (diagonal) preconditioner on GPU
    diag = cp.array(AtA.diagonal())
    diag_inv = cp.where(cp.abs(diag) > 1e-14, 1.0 / diag, 1.0)
    M = CpLinearOperator((ngrid, ngrid), matvec=lambda v: diag_inv * v)

    # Solve with GPU CG
    zgrid_flat_gpu, info = cp_cg(AtA_gpu, Atrhs_gpu, rtol=tol, maxiter=maxiter, M=M)

    if info > 0:
        import warnings
        warnings.warn(f"GPU CG did not converge within {maxiter} iterations (info={info}).")

    # Transfer back to CPU
    zgrid_flat = cp.asnumpy(zgrid_flat_gpu)
    zgrid = zgrid_flat.reshape(nx, ny).T

    return zgrid
