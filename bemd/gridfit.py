# REQ-BEMD-004: Surface fitting from scattered data
# Python implementation of gridfit using scipy sparse + regularization
# Original MATLAB gridfit by John D'Errico

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def gridfit(x, y, z, xnodes, ynodes, smoothness=1.0):
    """Estimates a surface on a 2D grid from scattered data.

    Uses a regularized least-squares approach with gradient smoothing.
    This is a simplified Python port of MATLAB gridfit by John D'Errico.

    Args:
        x: 1D array of x coordinates of scattered data.
        y: 1D array of y coordinates of scattered data.
        z: 1D array of z values at (x, y) points.
        xnodes: 1D array defining grid nodes in x direction.
        ynodes: 1D array defining grid nodes in y direction.
        smoothness: Smoothing parameter (default 1.0). Larger = smoother.

    Returns:
        zgrid: 2D array (len(ynodes) x len(xnodes)) containing the fitted surface.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    xnodes = np.asarray(xnodes, dtype=float).copy().ravel()
    ynodes = np.asarray(ynodes, dtype=float).copy().ravel()

    # Remove NaN data
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    n = len(x)
    if n < 3:
        raise ValueError("Insufficient data for surface estimation.")

    # Extend nodes to cover data range
    xnodes[0] = min(xnodes[0], x.min())
    xnodes[-1] = max(xnodes[-1], x.max())
    ynodes[0] = min(ynodes[0], y.min())
    ynodes[-1] = max(ynodes[-1], y.max())

    dx = np.diff(xnodes)
    dy = np.diff(ynodes)
    nx = len(xnodes)
    ny = len(ynodes)
    ngrid = nx * ny

    # Autoscale
    xscale = np.mean(dx)
    yscale = np.mean(dy)

    # Determine which cell each point lies in
    indx = np.searchsorted(xnodes, x, side='right') - 1
    indy = np.searchsorted(ynodes, y, side='right') - 1
    indx = np.clip(indx, 0, nx - 2)
    indy = np.clip(indy, 0, ny - 2)

    # Linear index: column-major storage matching MATLAB convention
    # ind = indy + ny * indx
    ind = indy + ny * indx

    # Normalized coordinates within each cell
    tx = np.clip((x - xnodes[indx]) / dx[indx], 0, 1)
    ty = np.clip((y - ynodes[indy]) / dy[indy], 0, 1)

    # Triangle interpolation
    k = tx > ty
    L = np.ones(n, dtype=int)
    L[k] = ny

    t1 = np.minimum(tx, ty)
    t2 = np.maximum(tx, ty)

    row_idx = np.repeat(np.arange(n), 3)
    col_idx = np.concatenate([ind, ind + ny + 1, ind + L])
    values = np.concatenate([1 - t2, t1, t2 - t1])

    A = sparse.csr_matrix((values, (row_idx, col_idx)), shape=(n, ngrid))
    rhs = z.copy()

    # Build gradient regularizer
    reg_parts = []

    # Y-direction (along ynodes, i.e., interior y-nodes for all x-nodes)
    if ny > 2:
        i_y, j_y = np.meshgrid(np.arange(nx), np.arange(1, ny - 1), indexing='ij')
        i_y = i_y.ravel()
        j_y = j_y.ravel()
        ind_y = j_y + ny * i_y
        dy1 = dy[j_y - 1] / yscale
        dy2 = dy[j_y] / yscale
        m_y = len(ind_y)

        row_y = np.repeat(np.arange(m_y), 3)
        col_y = np.concatenate([ind_y - 1, ind_y, ind_y + 1])
        val_y = np.concatenate([
            -2.0 / (dy1 * (dy1 + dy2)),
            2.0 / (dy1 * dy2),
            -2.0 / (dy2 * (dy1 + dy2))
        ])
        reg_parts.append(sparse.csr_matrix((val_y, (row_y, col_y)), shape=(m_y, ngrid)))

    # X-direction (along xnodes, i.e., interior x-nodes for all y-nodes)
    if nx > 2:
        i_x, j_x = np.meshgrid(np.arange(1, nx - 1), np.arange(ny), indexing='ij')
        i_x = i_x.ravel()
        j_x = j_x.ravel()
        ind_x = j_x + ny * i_x
        dx1 = dx[i_x - 1] / xscale
        dx2 = dx[i_x] / xscale
        m_x = len(ind_x)

        row_x = np.repeat(np.arange(m_x), 3)
        col_x = np.concatenate([ind_x - ny, ind_x, ind_x + ny])
        val_x = np.concatenate([
            -2.0 / (dx1 * (dx1 + dx2)),
            2.0 / (dx1 * dx2),
            -2.0 / (dx2 * (dx1 + dx2))
        ])
        reg_parts.append(sparse.csr_matrix((val_x, (row_x, col_x)), shape=(m_x, ngrid)))

    if reg_parts:
        Areg = sparse.vstack(reg_parts)
        nreg = Areg.shape[0]

        # Scale regularizer relative to interpolation equations
        NA = sparse.linalg.norm(A, ord=1)
        NR = sparse.linalg.norm(Areg, ord=1)
        if NR > 0:
            scale = smoothness * NA / NR
        else:
            scale = smoothness

        A_full = sparse.vstack([A, Areg * scale])
        rhs_full = np.concatenate([rhs, np.zeros(nreg)])
    else:
        A_full = A
        rhs_full = rhs

    # Solve using LSQR (more robust than normal equations for potentially
    # rank-deficient systems)
    result = lsqr(A_full, rhs_full, atol=1e-12, btol=1e-12,
                  iter_lim=max(10000, ngrid))
    zgrid_flat = result[0]

    # Reshape: the linear index is ind = iy + ny*ix,
    # so zgrid_flat stores nx columns of ny elements each.
    # reshape(nx, ny) gives [ix, iy], transpose gives [iy, ix] = (ny, nx)
    zgrid = zgrid_flat.reshape(nx, ny).T

    return zgrid
