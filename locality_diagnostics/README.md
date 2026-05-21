# Locality Diagnostics

This directory contains ASCAD temporal-window diagnostics for score-local and trace-local behavior.

- `ascad_temporal_window_sweep.py`: evaluates BI quantities across ASCAD temporal windows and neighborhood radii.

Run locally with:

```bash
python -m locality_diagnostics.ascad_temporal_window_sweep --help
```

The expected inputs are prepared ASCAD traces and local checkpoint or score artifacts. Write generated outputs under `results/locality_diagnostics/`.
