# REQ-BEMD-005: 2D sifting process for extracting a single IMF
# Ported from MATLAB sift function in bemd.m

import numpy as np
from bemd.extrema2 import extrema2
from bemd.gridfit import gridfit


def sift(input_image):
    """Sifts for a single IMF of a given 2D signal.

    Args:
        input_image: 2D numpy array.

    Returns:
        tuple: (h_imf, residue)
            h_imf   - extracted IMF component
            residue - residue after IMF extraction
    """
    rows, cols = input_image.shape
    x = np.arange(rows)
    y = np.arange(cols)
    input_image_temp = input_image.copy()
    h_imf = input_image_temp.copy()

    max_iterations = 100

    for iteration in range(max_iterations):
        # Find extrema in the 2D signal
        zmax, imax, zmin, imin = extrema2(input_image_temp)

        if len(zmax) < 3 or len(zmin) < 3:
            # Not enough extrema to interpolate; current state is the IMF
            h_imf = input_image_temp
            break

        xmax, ymax = np.unravel_index(imax, input_image_temp.shape)
        xmin, ymin = np.unravel_index(imin, input_image_temp.shape)

        # Interpolate extrema to get envelope surfaces using gridfit
        zmaxgrid = gridfit(ymax, xmax, zmax, y, x)
        zmingrid = gridfit(ymin, xmin, zmin, y, x)

        # Average envelope
        zavggrid = (zmaxgrid + zmingrid) / 2.0

        # Compute IMF candidate
        h_imf = input_image_temp - zavggrid

        # Handle NaN in result (can occur with sparse extrema)
        if np.any(np.isnan(h_imf)):
            h_imf = np.nan_to_num(h_imf, nan=0.0)

        # Compute stopping criterion (SD)
        eps = 1e-8
        num = np.sum((h_imf - input_image_temp) ** 2)
        den = np.sum(input_image_temp ** 2) + eps
        cost = num / den

        if cost < 0.2:
            break
        else:
            input_image_temp = h_imf

    # Compute residue
    residue = input_image - h_imf

    return h_imf, residue
