# Bayes Oracle Diagnostics

This directory contains synthetic known-posterior checks used to separate real-data challenger diagnostics from true Bayes-oracle validation.

- `gaussian_oracle_suite.py`: byte-scale Gaussian oracle suite audit.
- `bayes_oracle_synthetic.py`: masked/locality synthetic oracle audit with paper-style plots.

Run locally with:

```bash
python -m bayes_oracle.gaussian_oracle_suite --help
python -m bayes_oracle.bayes_oracle_synthetic --help
```

Write generated CSVs and figures to an ignored `results/` path.
