# REQ-BEMD-002: 1D extrema detection
# Ported from MATLAB extrema.m by Carlos Adrian Vargas Aguilera

import numpy as np


def extrema(x):
    """Gets the global extrema points from a 1D signal.

    Args:
        x: 1D numpy array.

    Returns:
        tuple: (xmax, imax, xmin, imin)
            xmax - maxima values in descending order
            imax - indexes of xmax
            xmin - minima values in ascending order
            imin - indexes of xmin
    """
    x = np.asarray(x, dtype=float).ravel()
    Nt = len(x)

    if Nt == 0:
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    # Handle NaN's
    inan = np.where(np.isnan(x))[0]
    indx = np.arange(Nt)
    if len(inan) > 0:
        mask = np.ones(Nt, dtype=bool)
        mask[inan] = False
        indx = indx[mask]
        x = x[mask]
        Nt = len(x)

    if Nt < 2:
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    # Difference between subsequent elements
    dx = np.diff(x)

    # Is it a horizontal line?
    if not np.any(dx):
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    # Flat peaks: put the middle element
    a = np.where(dx != 0)[0]
    if len(a) == 0:
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    lm = np.where(np.diff(a) != 1)[0] + 1
    if len(lm) > 0:
        d = a[lm] - a[lm - 1]
        a[lm] = a[lm] - d // 2
    a = np.append(a, Nt - 1)

    # Peaks detection
    xa = x[a]
    b = (np.diff(xa) > 0).astype(int)
    xb = np.diff(b)
    imax_local = np.where(xb == -1)[0] + 1
    imin_local = np.where(xb == 1)[0] + 1
    imax_local = a[imax_local]
    imin_local = a[imin_local]

    nmaxi = len(imax_local)
    nmini = len(imin_local)

    # Maximum or minimum on a flat peak at the ends?
    if nmaxi == 0 and nmini == 0:
        if x[0] > x[Nt - 1]:
            return (np.array([x[0]]), np.array([indx[0]]),
                    np.array([x[Nt - 1]]), np.array([indx[Nt - 1]]))
        elif x[0] < x[Nt - 1]:
            return (np.array([x[Nt - 1]]), np.array([indx[Nt - 1]]),
                    np.array([x[0]]), np.array([indx[0]]))
        else:
            return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    # Maximum or minimum at the ends?
    if nmaxi == 0:
        imax_local = np.array([0, Nt - 1])
    elif nmini == 0:
        imin_local = np.array([0, Nt - 1])
    else:
        if imax_local[0] < imin_local[0]:
            imin_local = np.concatenate(([0], imin_local))
        else:
            imax_local = np.concatenate(([0], imax_local))
        if imax_local[-1] > imin_local[-1]:
            imin_local = np.append(imin_local, Nt - 1)
        else:
            imax_local = np.append(imax_local, Nt - 1)

    xmax = x[imax_local]
    xmin = x[imin_local]

    # Restore original indices if NaN's were removed
    if len(inan) > 0:
        imax_local = indx[imax_local]
        imin_local = indx[imin_local]

    # Sort: maxima in descending order, minima in ascending order
    order_max = np.argsort(-xmax)
    xmax = xmax[order_max]
    imax_local = imax_local[order_max]

    order_min = np.argsort(xmin)
    xmin = xmin[order_min]
    imin_local = imin_local[order_min]

    return xmax, imax_local, xmin, imin_local
