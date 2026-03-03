#!/usr/bin/env python
# REQ-BEMD-008: Sample image evaluation for BEMD
# This script generates a synthetic test image with multiple frequency
# components and decomposes it using BEMD.

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from bemd import bemd


def create_sample_image(size=64):
    """Create a synthetic image with multiple frequency components.

    The image combines:
    - Low frequency: slow-varying sine wave (trend)
    - Medium frequency: medium-varying pattern
    - High frequency: fine texture
    """
    x, y = np.meshgrid(np.linspace(0, 4 * np.pi, size),
                       np.linspace(0, 4 * np.pi, size))

    # Low frequency component (trend)
    low_freq = 50 * np.sin(0.5 * x) * np.sin(0.5 * y)

    # Medium frequency component
    mid_freq = 30 * np.sin(2 * x) * np.cos(2 * y)

    # High frequency component (texture)
    high_freq = 15 * np.sin(5 * x + 1) * np.sin(5 * y + 1)

    # Combined signal
    combined = low_freq + mid_freq + high_freq + 128  # offset to positive range

    return combined, low_freq, mid_freq, high_freq


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "bemd_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BEMD Sample Image Evaluation")
    print("=" * 60)

    # Create sample image
    size = 64
    nimfs = 4
    print(f"\n1. Creating synthetic image ({size}x{size})...")
    combined, low_freq, mid_freq, high_freq = create_sample_image(size)
    print(f"   Signal range: [{combined.min():.1f}, {combined.max():.1f}]")

    # Save original components for comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(combined, cmap='gray')
    axes[0, 0].set_title("Combined (Input)")
    axes[0, 1].imshow(low_freq, cmap='RdBu_r')
    axes[0, 1].set_title("Low Freq Component (Ground Truth)")
    axes[1, 0].imshow(mid_freq, cmap='RdBu_r')
    axes[1, 0].set_title("Mid Freq Component (Ground Truth)")
    axes[1, 1].imshow(high_freq, cmap='RdBu_r')
    axes[1, 1].set_title("High Freq Component (Ground Truth)")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ground_truth.png"), dpi=150)
    plt.close(fig)
    print("   Saved: ground_truth.png")

    # Run BEMD
    print(f"\n2. Running BEMD (extracting {nimfs} IMFs)...")
    imf_matrix = bemd(combined, nimfs)

    # Visualize results
    print("\n3. Saving results...")
    fig, axes = plt.subplots(2, nimfs, figsize=(4 * nimfs, 8))

    for k in range(nimfs):
        imf_slice = imf_matrix[:, :, k]
        label = f"Residue" if k == nimfs - 1 else f"IMF {k + 1}"

        # Top row: IMF images
        im = axes[0, k].imshow(imf_slice, cmap='RdBu_r')
        axes[0, k].set_title(label)
        axes[0, k].axis("off")
        plt.colorbar(im, ax=axes[0, k], fraction=0.046)

        # Bottom row: histograms
        axes[1, k].hist(imf_slice.ravel(), bins=50, color='steelblue', alpha=0.7)
        axes[1, k].set_title(f"{label} Distribution")
        axes[1, k].set_xlabel("Value")
        axes[1, k].set_ylabel("Count")

        # Print statistics
        print(f"   {label}: mean={imf_slice.mean():.4f}, "
              f"std={imf_slice.std():.4f}, "
              f"range=[{imf_slice.min():.2f}, {imf_slice.max():.2f}]")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "bemd_results.png"), dpi=150)
    plt.close(fig)
    print("   Saved: bemd_results.png")

    # Reconstruction check
    print("\n4. Reconstruction check:")
    reconstructed = np.sum(imf_matrix[:, :, :-1], axis=2) + imf_matrix[:, :, -1]
    error = np.abs(reconstructed - combined)
    print(f"   Max reconstruction error: {error.max():.6e}")
    print(f"   Mean reconstruction error: {error.mean():.6e}")

    # Save individual IMFs as images
    for k in range(nimfs):
        imf_slice = imf_matrix[:, :, k]
        imf_min = imf_slice.min()
        imf_max = imf_slice.max()
        if imf_max - imf_min > 0:
            imf_norm = ((imf_slice - imf_min) / (imf_max - imf_min) * 255).astype(np.uint8)
        else:
            imf_norm = np.zeros_like(imf_slice, dtype=np.uint8)
        label = "residue" if k == nimfs - 1 else f"imf{k + 1}"
        Image.fromarray(imf_norm).save(os.path.join(output_dir, f"{label}.png"))

    print(f"\n5. All outputs saved to: {output_dir}/")
    print("=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
