# locality_diagnostics

**What this folder does.** It checks how much of a trace a trained network really uses. For each window length `tau`, the script keeps only a `tau`-sample window of every ASCAD trace at its original position (zeros everywhere else), picks a good window start, and records how often the network still guesses the key byte. The paper's full version plots that success rate against `tau` (`fig:tau-sweep`).

## Files

| File | Purpose |
|---|---|
| `ascad_temporal_window_sweep.py` | the window sweep for one pretrained CNN |

## Before you run

- TensorFlow (`requirements-ml.txt`) and `h5py`.
- An ASCAD file with `Profiling_traces` and `Attack_traces` groups. The TCHES20 checkout has one at `datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets/ASCAD_dataset/ASCAD.h5`.
- A pretrained Keras CNN for that file, for example one of the TCHES20 desync-0 models.

## Run

See every option:

```bash
python -m locality_diagnostics.ascad_temporal_window_sweep --help
```

A quick run (two window lengths, one seed, the center position only, about a minute):

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m locality_diagnostics.ascad_temporal_window_sweep \
  --ascad-h5 $T/datasets/ASCAD_dataset/ASCAD.h5 \
  --cnn-model $T/models/pretrained_models/models/zaid_ascad_desync_0_no_preprocessing_0.hdf5 \
  --tau-values 50,700 --n-seeds 1 --strategy center --output-dir results/locality_diagnostics
```

The defaults are the paper's sweep: `--tau-values 5,10,20,50,100,200,350,500,700`, five seeds, and `--strategy grid`, which searches over window positions on a subsample of the selection split. Labels are `Sbox[plaintext ^ key]` at `--target-byte` (default 2).

## Outputs

Written to `--output-dir` (default `results/tau_sweep`):

| File | Content |
|---|---|
| `tau_sweep_results.csv` | one row per (`tau`, seed); used to resume a stopped run |
| `tau_sweep_summary.csv` | the averages per `tau` |
| `figures/tau_sweep_{bi,success,positions}.{pdf,png}` | the plots |

`scripts/generate_paper_plots.py` reads `results/locality_diagnostics/tau_sweep_results.csv` by default (`--tau-results`).

## Notes

- The script pools the profiling and attack traces and splits the pool per seed into a holdout (`--holdout-size`, at most half the pool) and a position-selection set. Part of the holdout is therefore traces the CNN was trained on, so the success rate is not a fresh-trace estimate. The paper labels this figure a diagnostic on a mixed pool, not a certificate.
- The `bi_bits` columns and `figures/tau_sweep_bi.*` use a Hoeffding-style lower bound, `log2(K * (p_hat - sqrt(ln(1/delta) / (2 m))))` clipped to `[0, log2 K]`, with `M = 1`. The paper figure plots the KL-binomial interval instead, which `scripts/generate_paper_plots.py` recomputes from `holdout_success`, `n_holdout` and `n_positions_tested` (used as `M`).
- Earlier revisions of the script shifted the window to indices `0..tau-1` before zero-padding. CSV files from those revisions are not comparable with the current ones.
