# REQ-BEMD-001: Main BEMD (Bidimensional Empirical Mode Decomposition) function
# Ported from MATLAB bemd.m

import time
import numpy as np
from bemd.sift import sift


def bemd(input_image, nimfs):
    """Computes the Bidimensional EMD of a 2D signal.

    Decomposes a 2D signal (image) into a set of Intrinsic Mode Functions (IMFs)
    plus a residue, using the sifting process.

    Args:
        input_image: 2D numpy array (grayscale image or 2D signal).
        nimfs: Number of IMFs to extract.

    Returns:
        imf_matrix: 3D numpy array of shape (rows, cols, nimfs).
            The last slice contains the final residue.
    """
    start_time = time.time()

    input_image_d = np.asarray(input_image, dtype=float)
    rows, cols = input_image_d.shape

    imf_matrix = np.zeros((rows, cols, nimfs))
    h_func = input_image_d.copy()

    k = 0
    while k < nimfs - 1:
        print(f"  Extracting IMF {k + 1}/{nimfs}...")
        imf_temp, residue_temp = sift(h_func)
        imf_matrix[:, :, k] = imf_temp
        k += 1
        h_func = residue_temp

    # Assign the final residue to the last IMF index
    imf_matrix[:, :, k] = residue_temp

    elapsed = time.time() - start_time
    print(f"  BEMD completed in {elapsed:.2f} seconds.")

    return imf_matrix
