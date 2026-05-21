# BI Suite Evaluation

This directory contains evaluators for BI-suite rows that are computed from declared checkpoint suites:

- `ascad_random_key`: EstraNet checkpoint suite from `checkpoints/ascadr`
- `ches_ctf_2025`: EstraNet 3000-window checkpoint suite from `checkpoints/ches25_lx_w3000`

The evaluator computes single-target classification success on the attack split for every declared checkpoint. These success counts are the inputs to the KL-binomial `BI^{suite}` bracket. Key-rank and GE curves remain separate SCSCA/SASCA evidence and are not used as BI input.

Main Python entry points:

- `evaluate_estranet_bi_suite.py`: evaluate EstraNet checkpoint suites.
- `evaluate_tches20_pretrained_fixed_split.py`: evaluate TCHES20-style pretrained fixed-split suites.
- `summarize_ascad_estranet_bi_baselines.py`: summarize ASCAD EstraNet BI rows.
- `summarize_ches_transformer_bi_sweep.py`: summarize CHES Transformer BI rows.

Generated outputs should be written under a local `results/` directory or another ignored output path.
