---
title: "dimension_stability: what happens when traces get longer"
---

# What this folder shows

Figure 1 of the paper uses CHES-CTF-2025 traces cut to 100, 200, 500, 1,000, 3,000 and 7,000 points. For each length it plots BI next to three other information measures (a mutual-information estimate, PI through a small network, and PI through a Gaussian model). BI stays flat and above zero. The other three drift to zero or below as the traces get longer, because they struggle with many input points.

The BI line comes from the Transformer models of guide 01, Level C. The other three lines come from this folder.

- `ches_nonbi_metric_cell.py` computes one value: one length, one random seed, one measure. It is built for clusters: first it writes a list of numbered tasks, then each task is run by its number and adds one row to a results file.
- `real_nonbi_dimcap_baselines.py` and `real_projected_bi_dimcap.py` are related checks from the long version.
- `multivariate_stability_driver.py` runs the same measures on artificial traces and needs no data.

# Before you start

`CHES_Challenge.h5` in `datasets/raw/ches_ctf_2025/`, and TensorFlow for the network measure. One value takes seconds to minutes for the first two measures. The Gaussian measure is slow on long traces: about 8 minutes at 100 points, 40 minutes at 1,000, and 5 to 15 hours at 3,000 and 7,000. All 54 values behind Figure 1 take about 18 hours.

# Run: two quick values

Write the task list for two measures at 100 points:

```bash
python -m dimension_stability.ches_nonbi_metric_cell --write-manifest-only \
  --dimensions 100 --seeds 0 --metrics GKOV_MI,MLP_PI \
  --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
cat results/dimension_stability/manifest.csv
```

```{.text transcript="transcripts/07_dimension_stability/manifest.txt"}
```

Run task 0 (the mutual-information estimate):

```bash
python -m dimension_stability.ches_nonbi_metric_cell --task-id 0 \
  --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
```

```{.text transcript="transcripts/07_dimension_stability/task0.txt"}
```

Run task 1 (PI through a small network) and show the results file:

```bash
python -m dimension_stability.ches_nonbi_metric_cell --task-id 1 \
  --manifest results/dimension_stability/manifest.csv --output-dir results/dimension_stability
cat results/dimension_stability/ches_nonbi_highdim_checkpoint.csv
```

```{.text transcript="transcripts/07_dimension_stability/task1.txt"}
```

For the whole figure, write the task list with `--dimensions 100,200,500,1000,3000,7000 --seeds 0,1,2 --metrics ePI_or_PI,MLP_PI,GKOV_MI` and run the 54 tasks, ideally several at a time.

# Run: artificial traces

```bash
python -m dimension_stability.multivariate_stability_driver --dataset synthetic \
  --sweep-dimensions --dimensions 10,20,50 --n-seeds 2 --no-baselines \
  --output-dir results/dimension_stability/synthetic
```

```{.text transcript="transcripts/07_dimension_stability/synthetic.txt"}
```

# What to look for

- The results file has one row per value: length, seed, measure, the value in bits, and the run time. A finished value is skipped on rerun; `--force` recomputes it.
- The mutual-information and Gaussian values come out the same on every machine, to three decimals. The network value moves by a few tenths of a bit between seeds and machines, because the network is trained for only a few seconds; treat one value on its own as noise around zero.
- All three measures stay at or below zero while BI stays between 0.3 and 0.6 bits. That contrast is Figure 1.
