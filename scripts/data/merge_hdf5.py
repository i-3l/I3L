"""
Merge multiple HDF5 dataset files into a single file with renumbered demos.

Demos are copied in the order the source files are given, renamed to
demo_0..demo_{N-1}, and the data-group attrs (n_episodes, n_steps) are
recomputed. env_args and target_frequency_hz are carried over from the
sources and must agree across files.

Usage:
    # Explicit source files
    python merge_hdf5.py hang.hdf5 hang2.hdf5 hang3.hdf5 -o hang_50.hdf5

    # All files in a directory matching a task prefix (hang.hdf5, hang2.hdf5,
    # hang3.hdf5, ... sorted by their numeric suffix, bare name first)
    python merge_hdf5.py --prefix hang --data-dir ~/data -o hang_50.hdf5

    # Train/val split: first N demos go to -o, any extras go to --val-output
    # (train on exactly N demos; extras become the validation set)
    python merge_hdf5.py --prefix hang -o base_train/expert.hdf5 \
        --train-count 50 --val-output base_val/expert.hdf5
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
from tqdm import tqdm


def find_prefix_files(data_dir: Path, prefix: str) -> list[Path]:
    """Find <prefix>.hdf5, <prefix>2.hdf5/<prefix>_2.hdf5, ... by numeric suffix."""
    pattern = re.compile(rf"^{re.escape(prefix)}_?(\d*)\.hdf5$")
    matches = []
    for path in data_dir.iterdir():
        m = pattern.match(path.name)
        if m:
            suffix = int(m.group(1)) if m.group(1) else 0
            matches.append((suffix, path))
    return [path for _, path in sorted(matches)]


class _MergeTarget:
    """One output file accumulating renumbered demos."""

    def __init__(self, path: Path):
        if path.exists():
            sys.exit(f"Error: output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = h5py.File(path, "w")
        self.data_grp = self.file.create_group("data")
        self.idx = 0
        self.steps = 0

    def add(self, in_f: h5py.File, demo_name: str) -> None:
        in_f.copy(in_f["data"][demo_name], self.data_grp, name=f"demo_{self.idx}")
        self.steps += int(in_f["data"][demo_name].attrs["num_samples"])
        self.idx += 1

    def close(self, env_args, freq) -> None:
        if env_args is not None:
            self.data_grp.attrs["env_args"] = env_args
        if freq is not None:
            self.data_grp.attrs["target_frequency_hz"] = freq
        self.data_grp.attrs["n_episodes"] = self.idx
        self.data_grp.attrs["n_steps"] = self.steps
        self.file.close()
        size_gb = self.path.stat().st_size / 1e9
        print(f"Wrote {self.path}: {self.idx} demos, {self.steps} steps, {size_gb:.2f} GB")


def merge_files(
    sources: list[Path],
    output: Path,
    train_count: int | None = None,
    val_output: Path | None = None,
) -> None:
    train = _MergeTarget(output)
    val = None

    env_args = None
    freq = None
    total = 0
    for src in sources:
        with h5py.File(src, "r") as in_f:
            src_env_args = in_f["data"].attrs.get("env_args")
            src_freq = in_f["data"].attrs.get("target_frequency_hz")
            if env_args is not None and src_env_args != env_args:
                sys.exit(f"Error: env_args mismatch in {src}")
            if freq is not None and src_freq != freq:
                sys.exit(f"Error: target_frequency_hz mismatch in {src}")
            env_args, freq = src_env_args, src_freq

            demos = sorted(
                in_f["data"].keys(), key=lambda k: int(k.split("_")[-1])
            )
            for name in tqdm(demos, desc=src.name):
                if train_count is None or total < train_count:
                    train.add(in_f, name)
                else:
                    if val is None:
                        if val_output is None:
                            sys.exit(
                                f"Error: more than {train_count} demos but no "
                                "--val-output given for the extras"
                            )
                        val = _MergeTarget(val_output)
                    val.add(in_f, name)
                total += 1
        print(f"{src}: done, running total {total} demos")

    if train_count is not None and total < train_count:
        sys.exit(
            f"Error: only {total} demos found, fewer than --train-count {train_count}"
        )

    train.close(env_args, freq)
    if val is not None:
        val.close(env_args, freq)
    elif val_output is not None:
        print(f"No extra demos beyond {train_count}; no val file written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("sources", nargs="*", type=Path, help="source HDF5 files, merged in order")
    parser.add_argument("--prefix", help="task prefix to find <prefix>[N].hdf5 files in --data-dir")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data", help="directory searched with --prefix (default: ~/data)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output HDF5 path")
    parser.add_argument("--train-count", type=int, help="exact number of demos for --output; extras go to --val-output")
    parser.add_argument("--val-output", type=Path, help="output HDF5 path for demos beyond --train-count")
    args = parser.parse_args()

    if args.val_output and args.train_count is None:
        parser.error("--val-output requires --train-count")

    if args.prefix and args.sources:
        parser.error("give either source files or --prefix, not both")
    if args.prefix:
        sources = find_prefix_files(args.data_dir.expanduser(), args.prefix)
        if not sources:
            sys.exit(f"Error: no files matching {args.prefix}[N].hdf5 in {args.data_dir}")
    elif args.sources:
        sources = [p.expanduser() for p in args.sources]
    else:
        parser.error("give source files or --prefix")

    for src in sources:
        if not src.is_file():
            sys.exit(f"Error: source does not exist: {src}")

    print("Merging in order:")
    for src in sources:
        print(f"  {src}")
    merge_files(
        sources,
        args.output.expanduser(),
        train_count=args.train_count,
        val_output=args.val_output.expanduser() if args.val_output else None,
    )


if __name__ == "__main__":
    main()
