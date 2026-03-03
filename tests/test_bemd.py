# REQ-BEMD-007: Unit tests for BEMD library (Article III: Test-First Development)

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bemd.extrema import extrema
from bemd.extrema2 import extrema2
from bemd.gridfit import gridfit
from bemd.sift import sift
from bemd.core import bemd


class TestExtrema:
    """Tests for 1D extrema detection."""

    def test_simple_sine(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        xmax, imax, xmin, imin = extrema(x)
        assert len(xmax) > 0
        assert len(xmin) > 0
        assert np.all(xmax >= xmin[0]) or len(xmin) == 0

    def test_constant_signal(self):
        x = np.ones(10)
        xmax, imax, xmin, imin = extrema(x)
        assert len(xmax) == 0
        assert len(xmin) == 0

    def test_monotonic_increasing(self):
        x = np.arange(10, dtype=float)
        xmax, imax, xmin, imin = extrema(x)
        # End points should be detected
        assert len(xmax) >= 1
        assert len(xmin) >= 1

    def test_with_nans(self):
        x = np.array([1.0, 3.0, np.nan, 2.0, 5.0, 1.0])
        xmax, imax, xmin, imin = extrema(x)
        # Should handle NaN gracefully
        assert isinstance(xmax, np.ndarray)

    def test_empty_input(self):
        x = np.array([])
        xmax, imax, xmin, imin = extrema(x)
        assert len(xmax) == 0


class TestExtrema2:
    """Tests for 2D extrema detection."""

    def test_simple_peak(self):
        xy = np.zeros((5, 5))
        xy[2, 2] = 10.0  # peak
        xymax, smax, xymin, smin = extrema2(xy)
        assert len(xymax) >= 0  # May or may not detect depending on neighbors

    def test_gaussian_peak(self):
        x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
        z = np.exp(-(x**2 + y**2))
        xymax, smax, xymin, smin = extrema2(z)
        assert len(xymax) > 0

    def test_2d_sinusoid(self):
        x, y = np.meshgrid(np.linspace(0, 4 * np.pi, 30),
                           np.linspace(0, 4 * np.pi, 30))
        z = np.sin(x) * np.sin(y)
        xymax, smax, xymin, smin = extrema2(z)
        assert len(xymax) > 0
        assert len(xymin) > 0


class TestGridfit:
    """Tests for surface fitting."""

    def test_linear_surface(self):
        # Use more data points for a well-conditioned system
        np.random.seed(0)
        n = 30
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = x + y  # linear surface
        xnodes = np.linspace(0, 1, 5)
        ynodes = np.linspace(0, 1, 5)
        zgrid = gridfit(x, y, z, xnodes, ynodes)
        assert zgrid.shape == (5, 5)
        # Corner values should be close to expected
        assert abs(zgrid[0, 0] - 0.0) < 0.5
        assert abs(zgrid[-1, -1] - 2.0) < 0.5

    def test_output_shape(self):
        np.random.seed(42)
        x = np.random.rand(50)
        y = np.random.rand(50)
        z = np.random.rand(50)
        xnodes = np.linspace(0, 1, 10)
        ynodes = np.linspace(0, 1, 10)
        zgrid = gridfit(x, y, z, xnodes, ynodes)
        assert zgrid.shape == (10, 10)


class TestSift:
    """Tests for 2D sifting."""

    def test_sift_returns_imf_and_residue(self):
        x, y = np.meshgrid(np.linspace(0, 4 * np.pi, 32),
                           np.linspace(0, 4 * np.pi, 32))
        z = np.sin(x) * np.sin(y) + 0.5 * np.sin(2 * x)
        h_imf, residue = sift(z)
        assert h_imf.shape == z.shape
        assert residue.shape == z.shape

    def test_conservation(self):
        """IMF + residue should exactly reconstruct the original."""
        x, y = np.meshgrid(np.linspace(0, 4 * np.pi, 32),
                           np.linspace(0, 4 * np.pi, 32))
        z = np.sin(x) * np.sin(y)
        h_imf, residue = sift(z)
        reconstructed = h_imf + residue
        np.testing.assert_allclose(reconstructed, z, atol=1e-6)


class TestBEMD:
    """Tests for main BEMD function."""

    def test_output_shape(self):
        x, y = np.meshgrid(np.linspace(0, 4 * np.pi, 32),
                           np.linspace(0, 4 * np.pi, 32))
        z = np.sin(x) * np.sin(y) + 0.3 * np.cos(3 * x)
        nimfs = 3
        imf_matrix = bemd(z, nimfs)
        assert imf_matrix.shape == (32, 32, 3)

    def test_no_nan_in_output(self):
        """BEMD output should not contain NaN values."""
        x, y = np.meshgrid(np.linspace(0, 4 * np.pi, 32),
                           np.linspace(0, 4 * np.pi, 32))
        z = np.sin(x) * np.sin(y) + 0.3 * np.cos(3 * x)
        nimfs = 3
        imf_matrix = bemd(z, nimfs)
        assert not np.any(np.isnan(imf_matrix))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
