# REQ-BEMD-003: 2D extrema detection
# Ported from MATLAB extrema2.m by Carlos Adrian Vargas Aguilera

import numpy as np
from bemd.extrema import extrema


def extrema2(xy):
    """Gets the extrema points from a 2D surface.

    Searches through columns, rows, and diagonals of the matrix.

    Args:
        xy: 2D numpy array (M x N matrix).

    Returns:
        tuple: (xymax, smax, xymin, smin)
            xymax - maxima values in descending order
            smax  - linear indexes of xymax
            xymin - minima values in ascending order
            smin  - linear indexes of xymin
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2:
        raise ValueError("Entry must be a 2D matrix.")

    M, N = xy.shape

    # Search peaks through columns
    smaxcol, smincol = _extremos(xy)

    # Search peaks through rows, on columns with extrema points
    if len(smaxcol) > 0 or len(smincol) > 0:
        rows_with_col_extrema = []
        if len(smaxcol) > 0:
            rows_with_col_extrema.append(smaxcol[:, 0])
        if len(smincol) > 0:
            rows_with_col_extrema.append(smincol[:, 0])
        im = np.unique(np.concatenate(rows_with_col_extrema))
    else:
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    smaxfil, sminfil = _extremos(xy[im, :].T)

    # Convert from (row, col) to linear index
    if len(smaxcol) > 0:
        smaxcol_lin = np.ravel_multi_index((smaxcol[:, 0], smaxcol[:, 1]), (M, N))
    else:
        smaxcol_lin = np.array([], dtype=int)

    if len(smincol) > 0:
        smincol_lin = np.ravel_multi_index((smincol[:, 0], smincol[:, 1]), (M, N))
    else:
        smincol_lin = np.array([], dtype=int)

    if len(smaxfil) > 0:
        smaxfil_lin = np.ravel_multi_index((im[smaxfil[:, 1]], smaxfil[:, 0]), (M, N))
    else:
        smaxfil_lin = np.array([], dtype=int)

    if len(sminfil) > 0:
        sminfil_lin = np.ravel_multi_index((im[sminfil[:, 1]], sminfil[:, 0]), (M, N))
    else:
        sminfil_lin = np.array([], dtype=int)

    # Peaks in both rows and columns
    smax = np.intersect1d(smaxcol_lin, smaxfil_lin)
    smin = np.intersect1d(smincol_lin, sminfil_lin)

    # Search peaks through diagonals
    all_extrema = np.union1d(smax, smin)
    if len(all_extrema) > 0:
        iext, jext = np.unravel_index(all_extrema, (M, N))

        # Check peaks on down-up diagonal
        sextmax, sextmin = _extremos_diag(iext, jext, xy, 1)

        # Include corner points
        corners1 = np.array([np.ravel_multi_index((M - 1, 0), (M, N)),
                             np.ravel_multi_index((0, N - 1), (M, N))])
        smax = np.intersect1d(smax, np.union1d(corners1, sextmax))
        smin = np.intersect1d(smin, np.union1d(corners1, sextmin))

        # Peaks on up-down diagonals
        all_extrema2 = np.union1d(smax, smin)
        if len(all_extrema2) > 0:
            iext2, jext2 = np.unravel_index(all_extrema2, (M, N))
            sextmax2, sextmin2 = _extremos_diag(iext2, jext2, xy, -1)

            corners2 = np.array([0, M * N - 1])
            smax = np.intersect1d(smax, np.union1d(corners2, sextmax2))
            smin = np.intersect1d(smin, np.union1d(corners2, sextmin2))

    # Extrema values
    if len(smax) > 0:
        xymax = xy.ravel()[smax]
        order_max = np.argsort(-xymax)
        xymax = xymax[order_max]
        smax = smax[order_max]
    else:
        xymax = np.array([])
        smax = np.array([], dtype=int)

    if len(smin) > 0:
        xymin = xy.ravel()[smin]
        order_min = np.argsort(xymin)
        xymin = xymin[order_min]
        smin = smin[order_min]
    else:
        xymin = np.array([])
        smin = np.array([], dtype=int)

    return xymax, smax, xymin, smin


def _extremos(matriz):
    """Find peaks through columns (or rows if transposed input).

    Args:
        matriz: 2D numpy array.

    Returns:
        tuple: (smax, smin) - arrays of (row, col) index pairs.
    """
    smax_list = []
    smin_list = []

    for n in range(matriz.shape[1]):
        col = matriz[:, n]
        _, imaxfil, _, iminfil = extrema(col)
        if len(imaxfil) > 0:
            imaxcol = np.full(len(imaxfil), n)
            smax_list.append(np.column_stack([imaxfil, imaxcol]))
        if len(iminfil) > 0:
            imincol = np.full(len(iminfil), n)
            smin_list.append(np.column_stack([iminfil, imincol]))

    if smax_list:
        smax = np.vstack(smax_list)
    else:
        smax = np.empty((0, 2), dtype=int)

    if smin_list:
        smin = np.vstack(smin_list)
    else:
        smin = np.empty((0, 2), dtype=int)

    return smax, smin


def _extremos_diag(iext, jext, xy, A):
    """Find peaks through diagonals.

    Args:
        iext, jext: row and column indices of candidate extrema.
        xy: 2D numpy array.
        A: 1 for down-up diagonal, -1 for up-down diagonal.

    Returns:
        tuple: (sextmax, sextmin) - linear indices of extrema on diagonals.
    """
    M, N = xy.shape

    if A == -1:
        iext = M - iext - 1

    # Find where each diagonal crosses the top/left edge
    iini, jini = _cruce(iext, jext, 0, 0, M, N)
    # Unique starting points
    start_indices = np.ravel_multi_index((iini, jini), (M, N))
    unique_starts = np.unique(start_indices)
    iini, jini = np.unravel_index(unique_starts, (M, N))

    # Find where each diagonal crosses the bottom/right edge
    ifin, jfin = _cruce(iini, jini, M - 1, N - 1, M, N)

    sextmax_list = []
    sextmin_list = []

    for n in range(len(iini)):
        length = min(ifin[n] - iini[n], jfin[n] - jini[n]) + 1
        ises = np.arange(iini[n], iini[n] + length)
        jses = np.arange(jini[n], jini[n] + length)

        if A == -1:
            ises = M - ises - 1

        # Clip to valid range
        valid = (ises >= 0) & (ises < M) & (jses >= 0) & (jses < N)
        ises = ises[valid]
        jses = jses[valid]

        if len(ises) < 2:
            continue

        s = np.ravel_multi_index((ises, jses), (M, N))
        _, imax_d, _, imin_d = extrema(xy.ravel()[s])
        if len(imax_d) > 0:
            sextmax_list.append(s[imax_d])
        if len(imin_d) > 0:
            sextmin_list.append(s[imin_d])

    if sextmax_list:
        sextmax = np.concatenate(sextmax_list)
    else:
        sextmax = np.array([], dtype=int)

    if sextmin_list:
        sextmin = np.concatenate(sextmin_list)
    else:
        sextmin = np.array([], dtype=int)

    return sextmax, sextmin


def _cruce(i0, j0, I, J, M, N):
    """Find where diagonals cross the matrix boundary.

    Args:
        i0, j0: starting row and column indices.
        I, J: target boundary (0,0 for top-left; M-1,N-1 for bottom-right).
        M, N: matrix dimensions.

    Returns:
        tuple: (i, j) - boundary crossing indices.
    """
    i0 = np.asarray(i0)
    j0 = np.asarray(j0)

    if I == 0 and J == 0:
        # Going toward top-left
        offset = np.minimum(i0, j0)
        i = i0 - offset
        j = j0 - offset
    else:
        # Going toward bottom-right
        offset = np.minimum(I - i0, J - j0)
        i = i0 + offset
        j = j0 + offset

    return i.astype(int), j.astype(int)
