---
title: "core, datasets and scripts: the shared parts"
---

# core: turning counts into certificates

`bi_certificate.py` turns a count of correct guesses into a BI value. `baseline_metrics.py` has the other information measures (PI, HI and mutual information) used for comparison. `data_loader.py` finds and loads the trace files and can also make artificial traces. Every experiment imports these instead of carrying its own copy.

The loader has a self-test that needs no data:

```bash
python -m core.data_loader
```

```{.text transcript="transcripts/11_core_datasets_scripts/data_loader.txt"}
```

# datasets: where the traces live

All downloaded traces go under `datasets/raw/`, which git ignores. `setup_datasets.py` can fetch two trace sets that the other scripts do not: `--ascad` (the plain ASCAD file) and `--aes-hd`. It can also check which files are present:

```bash
python -m datasets.setup_datasets --info
python -m datasets.setup_datasets --verify --data-dir datasets/raw
```

```{.text transcript="transcripts/11_core_datasets_scripts/setup_datasets.txt"}
```

The check reports the two optional files as missing unless you fetched them; that is fine for all four levels.

# scripts: extra tables and plots

`plot_bi_loc_radius_curves.py` draws the curves of Figure 2. `generate_paper_plots.py` draws the extra plots of the long version from the results of guides 07 to 09; a few of its inputs come from runs whose scripts are not in this repository, and `scripts/README.md` lists them. The two figures of the paper do not depend on this folder.

```bash
python -m scripts.plot_bi_loc_radius_curves --help
```

```{.text transcript="transcripts/11_core_datasets_scripts/plot_help.txt"}
```

# docs

`docs/EXPERIMENT_MAP.md` lists every table and figure of the paper and the script that makes it.
