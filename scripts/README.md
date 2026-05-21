# Postprocessing And Plot Scripts

This directory contains table builders, summary scripts, and paper-plot generation code.

- `build_bi_suite_paper_artifacts.py`: builds finite-suite BI paper tables from seed rows and baseline summaries.
- `audit_bi_suite_margin_reduction.py`: audits margin-reduction rows used in suite tightness diagnostics.
- `build_full_suite_bi_loc_artifacts.py`: builds full-suite score-local BI artifacts.
- `build_audited_score_radius_points.py`: prepares audited score-radius points.
- `plot_bi_loc_radius_curves.py`: generates BI_loc radius plots.
- `generate_paper_plots.py`: regenerates the main paper diagnostic plots from current CSV artifacts.
- `build_real_challenger_diagnostic.py`: summarizes real-data out-of-family challenger diagnostics.
- `merge_ascad_fixed_split_mlp_challengers.py`: merges ASCAD fixed-split MLP challenger rows.

Paper plots are generated with:

```bash
python -m scripts.generate_paper_plots \
  --out-dir results/paper_plots/Figures
```

If you regenerate only local-family radius figures, use:

```bash
python -m scripts.plot_bi_loc_radius_curves --help
```

All generated tables and plots should be written under `results/` or another ignored output path.
