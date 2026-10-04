# core

**What this folder does.** The shared code that every experiment imports: the certificate arithmetic, the baseline information measures, and the data loading. Import these modules from experiment scripts instead of duplicating them.

## Files

| File | Purpose |
|---|---|
| `bi_certificate.py` | KL-binomial BI certificate routines and the conversion from success counts to bits |
| `baseline_metrics.py` | PI, HI, MI and related non-BI estimators |
| `data_loader.py` | dataset discovery, raw trace loading, synthetic trace generation and split helpers |

## Quick check

The loader has a self-test that needs no data: it reports which datasets it can find under `datasets/raw/`, generates a synthetic trace set, splits it, and builds four representations.

```bash
python -m core.data_loader
```
