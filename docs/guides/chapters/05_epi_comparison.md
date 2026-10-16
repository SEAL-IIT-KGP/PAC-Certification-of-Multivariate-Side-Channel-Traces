---
title: "epi_comparison: BI next to a common information measure"
---

# What this folder shows

Perceived information (PI) is a common way to score how much a model knows about the key. It is sensitive to how confident the model is, not just to whether its guesses are right. A corrected version, effective PI (EPI), first rescales the model's confidence.

This folder compares BI with EPI on the same frozen model. Its main table (quoted in Section 5.4) rescales the model's confidence by factors from 0.05 to 20. Raw PI changes by tens of bits. EPI does not change at all, and no guess changes, so the count behind BI does not change either. Level D checks this table.

`epi_head_to_head.py` works on one trace set: it picks the most accurate of the 80 models, fits EPI's one parameter on separate traces, reports PI, EPI and BI with error bars, then repeats EPI at nine rescaling factors. `summarize_epi_head_to_head.py` combines the results of the six trace sets.

# Before you start

The Docker image (or the Python environment with `requirements-ml.txt`) and the TCHES20 download.

# Run

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m epi_comparison.epi_head_to_head --dataset ascad_desync_0 \
  --dataset-dir $T/datasets --tches20-src-dir $T/src --models-dir $T/models \
  --output-dir results/epi_comparison/main
```

```{.text transcript="transcripts/05_epi_comparison/head_to_head.txt"}
```

```{.text transcript="transcripts/05_epi_comparison/temperature.txt"}
```

Run the other five trace sets the same way (`ascad_desync_50`, `ascad_desync_100`, `aes_hd`, `aes_rd`, `dpav4`). Level D leaves all six under `results/paper_results/recomputed/epi_comparison/main`, so you can also summarize those:

```bash
python -m epi_comparison.summarize_epi_head_to_head \
  --input-dir results/paper_results/recomputed/epi_comparison/main --output-dir results/epi_comparison/summary
cat results/epi_comparison/summary/summary.csv
```

```{.text transcript="transcripts/05_epi_comparison/summary.txt"}
```

# What to look for

In `temperature_stress.csv`, the `raw_pi` column swings widely across the rows (by 4 bits on ASCAD, by 27 bits on AES-RD), the `epi` column stays the same to eight decimals, and the last column says that every guess stayed the same. That is the point: BI and EPI do not care how confident the model is, only what it guesses. Raw PI does.
