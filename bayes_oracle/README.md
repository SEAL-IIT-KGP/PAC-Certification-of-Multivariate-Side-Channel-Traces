# bayes_oracle

Synthetic traces whose best possible (Bayes) success rate is known exactly, so the BI of a model suite can be compared with the truth. These produce the known-posterior rows of the full version's Bayes-completeness table.

## Files

| File | Purpose |
|---|---|
| `gaussian_oracle_suite.py` | each bit of the key byte leaks through its own Gaussian channel; the Bayes-optimal attacker is known in closed form |
| `bayes_oracle_synthetic.py` | masked oracle with a width and receptive-field sweep of MLP attackers |

## Run

Needs only `requirements.txt` and no data. The paper's settings are the defaults (about 20 seconds):

```bash
python -m bayes_oracle.gaussian_oracle_suite \
    --amplitude 3.0 --sigma 1.0 \
    --n-train 102400 --n-attack 100000 --n-seeds 3 \
    --output-dir results/bayes_gaussian_oracle_suite
python -m bayes_oracle.bayes_oracle_synthetic --help
```

Level A of `paper_results/` runs the Gaussian oracle and checks its two numbers: Bayes success 0.989577 and suite success 0.989503.

## Outputs

Under `--output-dir`: `gaussian_oracle_suite_rows.csv` (one row per seed and model), `gaussian_oracle_suite_summary.csv` (Bayes reference and suite), and `figures/gaussian_oracle_suite_bi.{pdf,png}`.
