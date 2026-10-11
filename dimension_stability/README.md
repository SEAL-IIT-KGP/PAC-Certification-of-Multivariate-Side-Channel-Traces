# dimension_stability

Compares BI with PI, HI and MI estimators as traces grow longer. The estimator lines of Figure 1 (GKOV MI, MLP-PI and Gaussian-template ePI/PI on CHES-CTF-2025 traces of 100 to 7,000 samples) come from here; the BI line comes from Level C of `paper_results/`.

## Files

| File | Purpose |
|---|---|
| `ches_nonbi_metric_cell.py` | one (dimension, seed, metric) cell on the CHES-CTF-2025 profiling traces |
| `real_nonbi_dimcap_baselines.py` | reduced-dimension PI/HI/MI baselines on the TCHES20 benchmarks |
| `real_projected_bi_dimcap.py` | projected-dimension BI certificates on real traces |
| `multivariate_stability_driver.py` | the same comparison on synthetic traces; no data needed |

The first three scripts run as task lists: `--write-manifest-only` writes numbered tasks, then each `--task-id N` adds one row to a checkpoint CSV. Finished cells are skipped; `--force` recomputes them.

## Run

Needs `CHES_Challenge.h5` in `datasets/raw/ches_ctf_2025/` and TensorFlow. Two quick cells (GKOV MI and MLP-PI at 100 samples):

```bash
python -m dimension_stability.ches_nonbi_metric_cell --write-manifest-only \
  --dimensions 100 --seeds 0 --metrics GKOV_MI,MLP_PI \
  --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
python -m dimension_stability.ches_nonbi_metric_cell --task-id 0 --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
python -m dimension_stability.ches_nonbi_metric_cell --task-id 1 --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
```

For the whole figure, use `--dimensions 100,200,500,1000,3000,7000 --seeds 0,1,2 --metrics ePI_or_PI,MLP_PI,GKOV_MI` and run the 54 tasks (about 18 hours in total; ePI/PI at 3,000 and 7,000 samples takes 5 and 15 hours per cell).

Synthetic traces, no data:

```bash
python -m dimension_stability.multivariate_stability_driver --dataset synthetic \
  --sweep-dimensions --dimensions 10,20,50 --n-seeds 2 --no-baselines --output-dir results/dimension_stability/synthetic
```

The TCHES20 scripts take `--tches20-dataset-dir` and `--tches20-src-dir` (see `datasets/README.md`).

## Outputs

`ches_nonbi_highdim_checkpoint.csv` and `ches_nonbi_highdim_summary.csv` under `--output-dir`, one row per cell (dimension, seed, metric, value in bits, runtime). `paper_results/figure1_dimension_sweep.py` uses these cells when they are present under `results/paper_results/recomputed/`.
