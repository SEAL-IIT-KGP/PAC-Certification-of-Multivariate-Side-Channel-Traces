# attacker_scope_diagnostics

**What this folder does.** A suite certificate covers only the models in the suite. These diagnostics train nested sets of models ("scopes") of growing capacity on a benchmark and watch whether the certified success keeps growing. When adding capacity no longer raises it, the declared scope is saturated for that benchmark. They produce the full version's `fig:scope-comparison` and the real-data rows of the Bayes-completeness table.

## Files

| File | Purpose |
|---|---|
| `attacker_scope_diagnostic.py` | the full driver over datasets, scopes 1 to 4 and seeds, with figures |
| `scope_cell_runner.py` | one (dataset, seed, scope) cell, appended to a checkpoint file; this is how the diagnostic is split over a cluster |
| `aesrd_scope_seed_runner.py` | the AES-RD budget-and-seed variant; appends to `results/attacker_scope_diagnostic/aesrd_budget_checkpoint.csv`, which `scripts/build_real_challenger_diagnostic.py` reads |
| `evaluate_ascad_d0_challenger_fixed_split.py` | re-evaluates ASCAD d0 challengers on a fixed split |

Scopes 1 and 2 are scikit-learn models (LDA, logistic regression, naive Bayes, ridge, MLPs). Scopes 3 and 4 add small PyTorch CNNs.

## Before you run

- `requirements-ml.txt` (PyTorch for scopes 3 and 4).
- The TCHES20 benchmark (`paper_results/fetch_tches20.sh`). For the TCHES20 dataset tags, `--dataset-dir` must be `<checkout>/datasets` and `--tches20-src-dir` must be `<checkout>/src`; the loader returns the profiling traces.

## Run

See every option:

```bash
python -m attacker_scope_diagnostics.attacker_scope_diagnostic --help
python -m attacker_scope_diagnostics.scope_cell_runner --help
python -m attacker_scope_diagnostics.aesrd_scope_seed_runner --help
python -m attacker_scope_diagnostics.evaluate_ascad_d0_challenger_fixed_split --help
```

One cell. With all 700 trace points a scope-1 cell takes hours on a CPU (six 256-class models); `--max-features 50` keeps the 50 most variable points and finishes in minutes:

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m attacker_scope_diagnostics.scope_cell_runner --dataset ascad_desync_0 --seed 0 --scope 1 --max-features 50 \
  --dataset-dir $T/datasets --tches20-src-dir $T/src --device cpu \
  --output-dir results/attacker_scope_diagnostic --checkpoint results/attacker_scope_diagnostic/scope_checkpoint.csv
```

The full driver takes its datasets as one quoted string: `--datasets "ascad_desync_0 aes_rd"`.

## Outputs

Under `results/attacker_scope_diagnostic/`: `scope_checkpoint.csv` and `scope_results.csv` (one row per dataset, seed and scope: number of models `M`, best success rate `p_max`, BI in bits), `scope_run_config.jsonl`, and the figures `attacker_scope_bi_by_scope`, `_bi_monotonicity`, `_metrics_by_scope_<dataset>` and `_suite_size_and_margin` as PDF and PNG.

## Notes

- These scripts are diagnostics of the declared scope. The paper's certificates always come from the fixed, declared suite (`bi_suite_8datasets/`, `bi_scope_certificates/`), never from the challengers trained here.
- In `evaluate_ascad_d0_challenger_fixed_split.py`, challengers are fit on 80% of the profiling traces. `p_fixed` is scored on the full profiling set, which includes those training traces, so it is in-sample. `p_old_holdout` is scored on disjoint traces and is the out-of-sample estimate.
