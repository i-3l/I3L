# Figure 3 intervention curves

Run from the I3L repository in the `behavior` conda environment:

```bash
python scripts/data/plot_subtask_intervention_timelines.py
```

The default tasks are Bread, Open Fridge and Pick Up Radio, Books, Turning on
Radio, and Popcorn. Open Fridge uses the recollected `or-*` datasets; Turning
on Radio uses `radio-*`. Both have four rounds of 25 episodes. The script reads their four
rounds of annotation JSONs under `outputs/annotation/wensi-ai` and the corresponding
cached LeRobot Parquet files. Outputs go to `outputs/annotation/figure3_reproduced`.

The recovered calculation is:

1. Identify contiguous `int_state` runs; discard intervention runs shorter than
   15 frames. All frames remain in the denominator.
2. Assign each frame `t` in an episode of length `T` to `floor(50 * t / T)`.
3. Pool intervention and total frame counts across episodes in each bin and
   compute `100 * intervention_frames / total_frames`.
4. Apply `scipy.ndimage.gaussian_filter1d(rate, sigma=3)` with its default
   reflection boundary handling and truncation.
5. Plot the 50 values at `np.linspace(0, 100, 50)`, matching the original rendering.

The original stage dividers use mean `100 * end_frame / T` from Round 4 for
Bread and Round 2 for Books and Popcorn. The original panel y-limits are 42, 22,
and 47 percent, respectively. These choices affect presentation, not the rates.

## Numerical verification

The archived source is `corgiTrax/2026_arxiv_IIIL_paper`, commit
`537204f8dda3714e2b4b1e98d2a086174c70bf81`, under
`figures/timelines/fig_timeline_{bread,books,popcorn}_bare.pdf`.

Given a checkout of that repository:

```bash
python scripts/data/reproduce_figure3_archived_curves.py --source-repo /path/to/original-paper
python scripts/data/plot_subtask_intervention_timelines.py \
  --reference-geometry outputs/annotation/figure3_exact/archived_curve_geometry.json
```

The first script extracts the published vector coordinates for comparison. The
second computes curves independently from local rollout data and verifies all
600 coordinates and the stage dividers. Maximum observed vertical discrepancy
is below `7.5e-7` percentage points, consistent with rounding of PDF coordinates.
Its `verification.json` records per-round errors and source PDF SHA-256 hashes.
The combined preview has a new layout; the curve coordinates reproduce the
archived plots to their stored precision.

## Updated six-task grid

```bash
python scripts/data/plot_subtask_intervention_timelines.py \
  --reference-geometry outputs/annotation/figure3_exact/archived_curve_geometry.json \
  --archived-strawberry outputs/annotation/figure3_exact/archived_curve_geometry.json \
  --output-dir outputs/annotation/figure3_updated
```

This places Turning on Radio in the former Hang Mug panel and updates Open
Fridge with `or-*`. Strawberry uses archived PDF coordinates without additional
smoothing because its raw rollouts are unavailable locally. The other five tasks
are calculated from Parquet data. The two radio panels use Round 4 subtask
boundaries. `provenance.json` records the dataset mapping; `verification.json`
checks the three tasks that still use the original data. The paper's existing
photo-and-arrow composite is separate from this curve grid.
