# locality_diagnostics

Measures how much of a trace a trained network uses. For each window length `tau`, every ASCAD trace keeps only a `tau`-sample window at its original position (zeros elsewhere), and the script records how often the network still finds the key byte. The full version plots this success rate against `tau`.

## Files

| File | Purpose |
|---|---|
| `ascad_temporal_window_sweep.py` | the window sweep for one pretrained CNN |

## Run

Needs `requirements-ml.txt`, an ASCAD file and a pretrained Keras CNN; the TCHES20 benchmark has both. A quick run (two window lengths, one seed, about a minute):

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m locality_diagnostics.ascad_temporal_window_sweep \
  --ascad-h5 $T/datasets/ASCAD_dataset/ASCAD.h5 \
  --cnn-model $T/models/pretrained_models/models/zaid_ascad_desync_0_no_preprocessing_0.hdf5 \
  --tau-values 50,700 --n-seeds 1 --strategy center --output-dir results/locality_diagnostics
```

The defaults are the paper's sweep: `--tau-values 5,10,20,50,100,200,350,500,700`, five seeds, and `--strategy grid` (a search over window positions). Labels are `Sbox[plaintext ^ key]` at `--target-byte` (default 2).

## Outputs

Under `--output-dir` (default `results/tau_sweep`):

| File | Content |
|---|---|
| `tau_sweep_results.csv` | one row per (`tau`, seed); also used to resume a stopped run |
| `tau_sweep_summary.csv` | averages per `tau` |
| `figures/tau_sweep_{bi,success,positions}.{pdf,png}` | the plots |

`scripts/generate_paper_plots.py` reads `tau_sweep_results.csv` (`--tau-results`).
