# Core Utilities

Shared code used by the experiment lanes:

- `bi_certificate.py`: KL-binomial BI certificate routines and success-to-BI conversions.
- `baseline_metrics.py`: PI, HI, MI, and related non-BI baseline estimators.
- `data_loader.py`: dataset discovery, raw trace loading, synthetic trace generation, and split helpers.

Quick checks:

```bash
python -m core.data_loader
```

Import these modules from experiment scripts rather than writing duplicate certificate or loader logic.
