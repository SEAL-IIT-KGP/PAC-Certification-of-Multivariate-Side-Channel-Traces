# Locality Diagnostics

ASCAD (desync 0) temporal-window diagnostic for one fixed pretrained CNN.

- `ascad_temporal_window_sweep.py`: for each window length tau (default `5,10,20,50,100,200,350,500,700`) and seed, keeps a tau-sample window of every trace at its original time positions (zeros elsewhere), picks the window start on a subsample of the selection split (`--strategy grid|center`), and records the CNN's hard-label success rate at that start on a random holdout. Earlier revisions shifted the window to indices `0..tau-1` before zero-padding, so CSVs from those revisions are not comparable.

Run locally with:

```bash
python -m locality_diagnostics.ascad_temporal_window_sweep --help
```

Inputs: `--ascad-h5` (ASCAD `.h5` with `Profiling_traces` and `Attack_traces`; needs `h5py`) and `--cnn-model` (pretrained Keras CNN; needs TensorFlow/Keras). Labels are `Sbox[plaintext ^ key]` at `--target-byte` (default 2).

Outputs go to `--output-dir` (default `results/tau_sweep`): `tau_sweep_results.csv` (one row per (tau, seed); used to resume), `tau_sweep_summary.csv`, and `figures/tau_sweep_{bi,success,positions}.{pdf,png}`. `scripts/generate_paper_plots.py` reads `results/locality_diagnostics/tau_sweep_results.csv` by default (`--tau-results`).

Caveats:

- The script pools the ASCAD profiling and attack traces, then splits that pool per seed into a holdout (`--holdout-size`, capped at half the pool) and a position-selection set. Part of the holdout is therefore profiling traces that the pretrained CNN was trained on, so holdout success is not a fresh-trace estimate. The paper labels this figure a diagnostic on a mixed profiling/attack pool, not an independent certificate.
- The `bi_bits` columns (and `figures/tau_sweep_bi.*`) are a Hoeffding lower-bound style value, `log2(K * (p_hat - sqrt(ln(1/delta) / (2 m))))` clipped to `[0, log2 K]`, with M=1. The paper figure instead plots the KL-binomial interval (`BI_obs`, `BI+`), which `scripts/generate_paper_plots.py` recomputes from `holdout_success`, `n_holdout` and `n_positions_tested` (used as M).
