#!/usr/bin/env python3
"""Plot all correction trajectories with takeover distribution.

Top row: percentage of trajectories in takeover at each timestep.
Remaining rows: one per joint, all traces overlaid (blue=normal, orange=takeover).

Usage:
  python scripts/plot_takeover_distribution.py data/a1_pnp_hard/corrections_train/expert.hdf5
  python scripts/plot_takeover_distribution.py data/a1_pnp_hard/corrections_train/expert.hdf5 --out results/takeover_dist.png
"""

import argparse
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

JOINT_NAMES = ["Joint 0", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Gripper"]
C_NORMAL = "#4C72B0"
C_TAKEOVER = "#DD8452"


def load_demos(hdf5_path):
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            action = np.array(g["action"])
            if "policy/is_oracle_active" in g:
                is_oracle = np.array(g["policy/is_oracle_active"]).flatten().astype(bool)
            else:
                is_oracle = np.zeros(len(action), dtype=bool)
            demos.append({"action": action, "is_oracle_active": is_oracle})
    return demos


def plot_takeover_distribution(demos, out_path, title=None):
    n_joints = min(demos[0]["action"].shape[1], 7)

    # Normalized time axis: resample everything to N_BINS between 0 and 1
    N_BINS = 200
    t_norm = np.linspace(0, 1, N_BINS)

    # Compute takeover percentage at each normalized timestep
    takeover_count = np.zeros(N_BINS)
    for d in demos:
        T = len(d["action"])
        oracle = d["is_oracle_active"][:T].astype(float)
        t_orig = np.linspace(0, 1, T)
        resampled = np.interp(t_norm, t_orig, oracle)
        takeover_count += (resampled > 0.5).astype(float)

    takeover_pct = takeover_count / len(demos) * 100

    n_rows = 1 + n_joints
    height_ratios = [1.2] + [1] * n_joints
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.5 * n_rows),
                             sharex=True, gridspec_kw={"height_ratios": height_ratios})

    # Top row: takeover percentage
    ax = axes[0]
    ax.fill_between(t_norm, takeover_pct, color=C_TAKEOVER, alpha=0.4)
    ax.plot(t_norm, takeover_pct, color=C_TAKEOVER, linewidth=1.2)
    ax.set_ylabel("% Takeover", fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(100, takeover_pct.max() * 1.3 + 5))
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)
    plot_title = title or f"Takeover Distribution ({len(demos)} trajectories)"
    ax.set_title(plot_title, fontsize=12, fontweight="bold")

    # Joint rows
    for j in range(n_joints):
        ax = axes[1 + j]
        for d in demos:
            action = d["action"][:, j]
            oracle = d["is_oracle_active"]
            T = len(action)
            t_orig = np.linspace(0, 1, T)

            # Plot normal segments in blue
            normal_action = action.copy()
            normal_action[oracle] = np.nan
            ax.plot(t_orig, normal_action, color=C_NORMAL, alpha=0.25, linewidth=0.6)

            # Plot takeover segments in orange
            takeover_action = action.copy()
            takeover_action[~oracle] = np.nan
            ax.plot(t_orig, takeover_action, color=C_TAKEOVER, alpha=0.5, linewidth=1.0)

        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"Joint {j}"
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.plot([], color=C_NORMAL, linewidth=1.5, alpha=0.7, label="Normal")
            ax.plot([], color=C_TAKEOVER, linewidth=1.5, alpha=0.7, label="Takeover")
            ax.legend(fontsize=9, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Normalized Time (0–1)", fontsize=10)
    axes[-1].set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def find_correction_hdf5s(data_dir):
    """Find all expert.hdf5 files that contain correction data (policy/is_oracle_active)."""
    import h5py
    data_dir = Path(data_dir)
    results = []
    for p in sorted(data_dir.rglob("expert.hdf5")):
        try:
            with h5py.File(str(p), "r") as f:
                keys = list(f["data"].keys())
                if keys and "policy/is_oracle_active" in f["data"][keys[0]]:
                    results.append(p)
        except Exception:
            pass
    return results


def name_from_path(hdf5_path, data_dir):
    """Derive a human-readable name from the HDF5 path relative to data_dir."""
    rel = Path(hdf5_path).relative_to(Path(data_dir))
    parts = list(rel.parts[:-1])  # drop 'expert.hdf5'
    if not parts:
        return "corrections"
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Plot takeover distribution across correction trajectories")
    parser.add_argument("input", type=str, help="Path to expert.hdf5 or directory to scan for all correction HDF5s")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument("--out", type=str, default=None, help="Output path (single file mode only)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file mode
        out_path = args.out or str(out_dir / "takeover_distribution.png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        demos = load_demos(str(input_path))
        print(f"Loaded {len(demos)} demos from {input_path}")
        n_with_takeover = sum(1 for d in demos if d["is_oracle_active"].sum() > 0)
        print(f"  {n_with_takeover}/{len(demos)} have takeover")
        plot_takeover_distribution(demos, out_path, title=f"Takeover Distribution ({len(demos)} trajectories)")
    elif input_path.is_dir():
        # Directory mode: find all correction HDF5s
        hdf5s = find_correction_hdf5s(input_path)
        if not hdf5s:
            print(f"No correction HDF5 files found in {input_path}")
            return
        print(f"Found {len(hdf5s)} correction datasets in {input_path}\n")
        for hdf5_path in hdf5s:
            name = name_from_path(hdf5_path, input_path)
            demos = load_demos(str(hdf5_path))
            n_with_takeover = sum(1 for d in demos if d["is_oracle_active"].sum() > 0)
            print(f"[{name}] {len(demos)} demos, {n_with_takeover} with takeover")
            if n_with_takeover == 0:
                print(f"  Skipping (no takeover data)")
                continue
            out_path = str(out_dir / f"takeover_dist_{name}.png")
            title = f"{name} — Takeover Distribution ({len(demos)} traj, {n_with_takeover} with takeover)"
            plot_takeover_distribution(demos, out_path, title=title)
    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
