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

The "High-BI Gaussian byte oracle" row of the paper's synthetic Bayes-completeness table (Bayes success 0.989577, suite success 0.989503) is reproduced by the current defaults, spelled out here:

```bash
python -m bayes_oracle.gaussian_oracle_suite \
    --amplitude 3.0 --sigma 1.0 \
    --n-train 102400 --n-attack 100000 --n-seeds 3 \
    --output-dir results/bayes_gaussian_oracle_suite
```

As a sanity check, the analytic Bayes success for this setting is (1 - Phi(-3))^8 ≈ 0.98925. Earlier defaults (amplitude 2.0, n_attack 50000) give Bayes success ≈ 0.83 and do not match the paper row.
