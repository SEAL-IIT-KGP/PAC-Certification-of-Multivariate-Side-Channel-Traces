# attacker_scope_diagnostics

Trains nested sets of models ("scopes") of growing capacity on one benchmark and checks whether the certified success keeps rising. When added capacity no longer raises it, the scope is saturated for that benchmark. These produce the full version's scope-comparison figure and the real-data rows of the Bayes-completeness table. The paper's certificates always come from the fixed suites in `bi_suite_8datasets/` and `bi_scope_certificates/`.

## Files

| File | Purpose |
|---|---|
| `attacker_scope_diagnostic.py` | full driver over datasets, scopes 1–4 and seeds, with figures |
| `scope_cell_runner.py` | one (dataset, seed, scope) cell, appended to a checkpoint file |
| `aesrd_scope_seed_runner.py` | AES-RD variant with a training budget (read by `scripts/build_real_challenger_diagnostic.py`) |
| `evaluate_ascad_d0_challenger_fixed_split.py` | re-evaluates ASCAD d0 challengers on a fixed split |

Scopes 1 and 2 are scikit-learn models (LDA, logistic regression, naive Bayes, ridge, MLPs); scopes 3 and 4 add small PyTorch CNNs.

## Run

Needs `requirements-ml.txt` and the TCHES20 benchmark (`paper_results/fetch_tches20.sh`). One cell with the 50 most variable trace points (a few minutes; all 700 points take hours):

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m attacker_scope_diagnostics.scope_cell_runner --dataset ascad_desync_0 --seed 0 --scope 1 --max-features 50 \
  --dataset-dir $T/datasets --tches20-src-dir $T/src --device cpu \
  --output-dir results/attacker_scope_diagnostic --checkpoint results/attacker_scope_diagnostic/scope_checkpoint.csv
```

The full driver takes its datasets as one quoted string, for example `--datasets "ascad_desync_0 aes_rd"`. Every script lists its options with `--help`.

## Outputs

Under `results/attacker_scope_diagnostic/`:

- `scope_checkpoint.csv`, `scope_results.csv`: one row per dataset, seed and scope (number of models `M`, best success rate `p_max`, BI in bits);
- `scope_run_config.jsonl`;
- figures `attacker_scope_bi_by_scope`, `_bi_monotonicity`, `_metrics_by_scope_<dataset>`, `_suite_size_and_margin` (PDF and PNG).
