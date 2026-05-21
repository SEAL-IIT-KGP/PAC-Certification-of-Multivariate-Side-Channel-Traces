# Attacker Scope Diagnostics

This directory contains trained-attacker and challenger diagnostics used to audit whether the declared attacker suite is saturated.

- `attacker_scope_diagnostic.py`: main attacker-scope comparison driver.
- `scope_cell_runner.py`: single-cell wrapper around the main scope diagnostic.
- `aesrd_scope_seed_runner.py`: AES-RD budget and seed diagnostic wrapper.
- `evaluate_ascad_d0_challenger_fixed_split.py`: fixed-split ASCAD challenger evaluation.

Typical entry points:

```bash
python -m attacker_scope_diagnostics.attacker_scope_diagnostic --help
python -m attacker_scope_diagnostics.scope_cell_runner --help
python -m attacker_scope_diagnostics.aesrd_scope_seed_runner --help
python -m attacker_scope_diagnostics.evaluate_ascad_d0_challenger_fixed_split --help
```

These scripts are diagnostics for declared attacker scope. Final certificates should still be computed from the fixed, declared suite.
