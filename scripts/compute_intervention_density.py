#!/usr/bin/env python3
"""Compute intervention density histogram from correction data.

Outputs a numpy array of shape (N_BINS,) representing the sampling weight
per normalized timestep bin. Used by UpweightedSpecialistDataset.

Usage:
  python scripts/compute_intervention_density.py \
    --correction-hdf5 data/a1_pnp_hard_remote/correction1/corrections_train/expert.hdf5 \
    --output data/a1_pnp_hard/binary_tarjan/intervention_density.npy \
    --n-bins 50 --upweight-factor 10.0 --sigma 2.0
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def compute_density(hdf5_path, n_bins=50, sigma=2.0, upweight_factor=10.0):
    counts = np.zeros(n_bins)

    with h5py.File(hdf5_path, "r") as f:
        for dk in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])):
            demo = f["data"][dk]
            if "policy" not in demo or "int_state" not in demo["policy"]:
                continue
            int_state = np.array(demo["policy"]["int_state"][:]).flatten()
            T = len(int_state)
            if T == 0:
                continue

            intervention_mask = (int_state == 2)
            if not intervention_mask.any():
                continue

            # Normalized timesteps [0, 1] for intervention steps
            intervention_times = np.where(intervention_mask)[0] / T
            # Bin them
            bin_indices = np.clip((intervention_times * n_bins).astype(int), 0, n_bins - 1)
            for b in bin_indices:
                counts[b] += 1

    # Smooth with Gaussian
    smoothed = gaussian_filter1d(counts.astype(float), sigma=sigma)

    # Convert to weights: min=1.0, max=upweight_factor
    if smoothed.max() > 0:
        weights = 1.0 + (upweight_factor - 1.0) * (smoothed / smoothed.max())
    else:
        weights = np.ones(n_bins)

    return weights, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction-hdf5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--upweight-factor", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    weights, raw_counts = compute_density(
        args.correction_hdf5, args.n_bins, args.sigma, args.upweight_factor
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, weights)
    print(f"Saved intervention density weights to {args.output}")
    print(f"  Shape: {weights.shape}")
    print(f"  Min weight: {weights.min():.2f}, Max weight: {weights.max():.2f}")
    print(f"  Raw intervention counts per bin: min={raw_counts.min():.0f}, max={raw_counts.max():.0f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        x = np.linspace(0, 1, args.n_bins)
        ax1.bar(x, raw_counts, width=1.0/args.n_bins, alpha=0.7, color="red")
        ax1.set_ylabel("Intervention count")
        ax1.set_title("Intervention Density Histogram")

        ax2.plot(x, weights, color="#2166AC", linewidth=2)
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Sampling weight")
        ax2.set_xlabel("Normalized trajectory time (0=start, 1=end)")
        ax2.set_title(f"Upweighting curve (factor={args.upweight_factor}, sigma={args.sigma})")

        plt.tight_layout()
        plot_path = args.output.replace(".npy", ".png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
