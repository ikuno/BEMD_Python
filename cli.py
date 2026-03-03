#!/usr/bin/env python
# REQ-BEMD-006: CLI entry point for BEMD (Article II: CLI Interface Mandate)

import argparse
import sys

import numpy as np
from PIL import Image

from bemd import bemd


def main():
    parser = argparse.ArgumentParser(
        description="BEMD - Bidimensional Empirical Mode Decomposition"
    )
    parser.add_argument("input", help="Path to input image file")
    parser.add_argument(
        "-n", "--nimfs", type=int, default=3,
        help="Number of IMFs to extract (default: 3)"
    )
    parser.add_argument(
        "-o", "--output", default="bemd_output",
        help="Output directory for IMF images (default: bemd_output)"
    )
    args = parser.parse_args()

    # Load image as grayscale
    try:
        img = Image.open(args.input).convert("L")
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    input_array = np.array(img, dtype=float)
    print(f"Input image: {args.input} ({input_array.shape[0]}x{input_array.shape[1]})")
    print(f"Extracting {args.nimfs} IMFs...")

    imf_matrix = bemd(input_array, args.nimfs)

    # Save output
    import os
    os.makedirs(args.output, exist_ok=True)

    for k in range(args.nimfs):
        imf_slice = imf_matrix[:, :, k]
        # Normalize to 0-255 for visualization
        imf_min = imf_slice.min()
        imf_max = imf_slice.max()
        if imf_max - imf_min > 0:
            imf_norm = ((imf_slice - imf_min) / (imf_max - imf_min) * 255).astype(np.uint8)
        else:
            imf_norm = np.zeros_like(imf_slice, dtype=np.uint8)

        label = "residue" if k == args.nimfs - 1 else f"imf{k + 1}"
        out_path = os.path.join(args.output, f"{label}.png")
        Image.fromarray(imf_norm).save(out_path)
        print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
