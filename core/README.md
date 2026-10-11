# core

Shared code that every experiment imports.

| File | Purpose |
|---|---|
| `bi_certificate.py` | KL-binomial certificate and conversion from success counts to bits |
| `baseline_metrics.py` | PI, HI and MI estimators |
| `data_loader.py` | dataset discovery and loading, synthetic traces, splits |

The loader has a self-test that needs no data. It lists the datasets found under `datasets/raw/` and generates and splits a synthetic trace set:

```bash
python -m core.data_loader
```
