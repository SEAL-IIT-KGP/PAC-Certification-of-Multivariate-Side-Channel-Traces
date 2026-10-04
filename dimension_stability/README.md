# dimension_stability

**What this folder does.** It compares the stability of BI with PI, HI and MI-style estimators as the trace dimension grows. The estimator cells of Figure 1 (GKOV mutual information, MLP perceived information, Gaussian-template ePI/PI on CHES-CTF-2025 traces of 100 to 7,000 samples) come from here; the BI line of the figure comes from the EstraNet window suites (`paper_results/`, Level C).

## Files

| File | Purpose |
|---|---|
| `ches_nonbi_metric_cell.py` | one (dimension, seed, metric) cell on the CHES-CTF-2025 profiling traces |
| `real_nonbi_dimcap_baselines.py` | reduced-dimension non-BI baselines on the TCHES20 benchmarks |
| `real_projected_bi_dimcap.py` | projected-dimension BI certificates on real datasets |
| `multivariate_stability_driver.py` | synthetic multivariate stability driver; no data needed |

The first three scripts are written for job arrays: one run with `--write-manifest-only` writes a manifest of numbered tasks, then each task is run with `--task-id N` (or the `ARRAY_TASK_ID` environment variable) and appends its row to a checkpoint CSV under a file lock. A cell is skipped only if the checkpoint already has a `status=complete` row for it; failed cells are rerun; `--force` reruns completed cells too.

## Before you run

- For the CHES cells: `CHES_Challenge.h5` at `datasets/raw/ches_ctf_2025/`, and TensorFlow for the MLP-PI metric.
- For the TCHES20 baselines: the checkout, passed as `--tches20-dataset-dir` and `--tches20-src-dir` (see `datasets/README.md`).
- Runtime per cell: GKOV MI and MLP-PI seconds to minutes; ePI/PI about 8 minutes at 100 samples, 40 minutes at 1,000, and 5 and 15 hours at 3,000 and 7,000 (single-threaded). All 54 cells behind Figure 1 take about 18 hours.

## Run

```bash
python -m dimension_stability.ches_nonbi_metric_cell --help
python -m dimension_stability.real_nonbi_dimcap_baselines --help
python -m dimension_stability.real_projected_bi_dimcap --help
python -m dimension_stability.multivariate_stability_driver --help
```

Two quick CHES cells (GKOV MI and MLP-PI at 100 samples, seed 0):

```bash
python -m dimension_stability.ches_nonbi_metric_cell --write-manifest-only \
  --dimensions 100 --seeds 0 --metrics GKOV_MI,MLP_PI \
  --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
python -m dimension_stability.ches_nonbi_metric_cell --task-id 0 --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
python -m dimension_stability.ches_nonbi_metric_cell --task-id 1 --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
```

The whole figure: write the manifest with `--dimensions 100,200,500,1000,3000,7000 --seeds 0,1,2 --metrics ePI_or_PI,MLP_PI,GKOV_MI` and run the 54 tasks. The synthetic driver without data:

```bash
python -m dimension_stability.multivariate_stability_driver --dataset synthetic \
  --sweep-dimensions --dimensions 10,20,50 --n-seeds 2 --no-baselines --output-dir results/dimension_stability/synthetic
```

## Outputs

`ches_nonbi_highdim_checkpoint.csv` and `ches_nonbi_highdim_summary.csv` under `--output-dir` (one row per cell: dimension, seed, metric, value in bits, status, runtime). `paper_results/figure1_dimension_sweep.py` uses recomputed cells whenever a checkpoint is present under `results/paper_results/recomputed/`.

## Notes

- `multivariate_stability_driver.py`: `--include-baselines/--no-include-baselines` (`--no-baselines` also disables them) and `--resume/--no-resume` both default to on. With `--sweep-dimensions` the figures are the pooled per-dimension coefficient-of-variation and success-rate plots; the per-dataset bar chart is produced only by non-sweep runs with more than one dataset.
- GKOV MI and ePI/PI are deterministic across machines to three decimals. MLP-PI is a small network trained for seconds per cell and moves by tenths of a bit between seeds and machines.
