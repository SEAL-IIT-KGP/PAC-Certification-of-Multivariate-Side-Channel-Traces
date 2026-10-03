# Attacker Scope Diagnostics

This directory contains trained-attacker and challenger diagnostics used to audit whether the declared attacker suite is saturated.

- `attacker_scope_diagnostic.py`: main attacker-scope comparison driver.
- `scope_cell_runner.py`: single-cell wrapper around the main scope diagnostic.
- `aesrd_scope_seed_runner.py`: AES-RD budget and seed diagnostic wrapper. By default it appends to `results/attacker_scope_diagnostic/aesrd_budget_checkpoint.csv` (the file read by `scripts/build_real_challenger_diagnostic.py`), not to the main driver's `scope_checkpoint.csv`.
- `evaluate_ascad_d0_challenger_fixed_split.py`: ASCAD d0 challenger re-evaluation. Challengers are fit on 80% of the profiling traces. `p_fixed` is scored on the full profiling set, which includes those training traces, so it is in-sample and not a valid out-of-sample challenger estimate. `p_old_holdout` is scored on traces disjoint from training and is the out-of-sample estimate.

Typical entry points:

```bash
python -m attacker_scope_diagnostics.attacker_scope_diagnostic --help
python -m attacker_scope_diagnostics.scope_cell_runner --help
python -m attacker_scope_diagnostics.aesrd_scope_seed_runner --help
python -m attacker_scope_diagnostics.evaluate_ascad_d0_challenger_fixed_split --help
```

These scripts are diagnostics for declared attacker scope. Final certificates should still be computed from the fixed, declared suite.
