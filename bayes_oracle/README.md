# bayes_oracle

**What this folder does.** On real traces nobody knows the best possible success rate. These scripts build synthetic traces whose posterior is known by construction, so the BI of a declared suite can be compared with the true Bayes success. They produce the "known-posterior oracle" rows of the full version's Bayes-completeness table.

## Files

| File | Purpose |
|---|---|
| `gaussian_oracle_suite.py` | byte-scale Gaussian oracle: each bit of the key byte leaks through its own Gaussian channel, so the Bayes-optimal attacker is known in closed form |
| `bayes_oracle_synthetic.py` | masked, locality-structured oracle with a width and receptive-field sweep of MLP attackers (PyTorch optional) |

## Before you run

Only `requirements.txt`. No data. The Gaussian oracle takes about twenty seconds. `paper_results/reproduce.py archived` runs it first and checks its two numbers.

## Run

```bash
python -m bayes_oracle.gaussian_oracle_suite --help
python -m bayes_oracle.bayes_oracle_synthetic --help
```

The paper row ("High-BI Gaussian byte oracle": Bayes success 0.989577, suite success 0.989503) comes from the defaults, spelled out here:

```bash
python -m bayes_oracle.gaussian_oracle_suite \
    --amplitude 3.0 --sigma 1.0 \
    --n-train 102400 --n-attack 100000 --n-seeds 3 \
    --output-dir results/bayes_gaussian_oracle_suite
```

## Outputs

`gaussian_oracle_suite_rows.csv` (one row per seed and model), `gaussian_oracle_suite_summary.csv` (the Bayes reference and the suite), a `README.md` and `figures/gaussian_oracle_suite_bi.{pdf,png}`, all under `--output-dir`.

## Notes

- The analytic Bayes success for amplitude 3 and unit noise is `(1 - Phi(-3))^8 ≈ 0.98925`.
- Earlier defaults (amplitude 2, 50,000 attack traces) give a Bayes success of about 0.83 and do not match the paper row.
