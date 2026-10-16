# multibyte_composition

Recovers four key bytes of the fixed-key ASCAD capture from the full 100,000-sample raw traces. This is the full version's four-byte table, quoted in Section 5.3 and checked by Level D of `paper_results/`.

## Files, in the order they run

| Step | File | What it does |
|---|---|---|
| 1 | `fetch_raw_ascad.py` | downloads `ASCAD_data.zip` (4.4 GB), extracts `ATMega8515_raw_traces.h5` (6 GB) and verifies both; `--zip PATH` reuses an archive, `--keep-zip` keeps it |
| 2 | `scan_raw_ascad_pois.py` | finds the 20 most correlated sample positions per byte and leakage label (about 7 min on 8 cores) |
| 3 | `train_raw_ascad_byte_attacker.py --target-byte B` | trains a two-layer MLP for one byte on windows around those positions and stores its attack-split posteriors |
| 4 | `compose_four_byte_recovery.py` | picks the four best bytes and counts the campaigns in which the full four-byte key ranks first |

## Run

Needs `requirements-ml.txt` (PyTorch) and 11 GB of free disk while extracting:

```bash
python -m multibyte_composition.fetch_raw_ascad
python -m multibyte_composition.scan_raw_ascad_pois
for b in 0 1 2 3 4 5 8 15; do python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte $b; done
python -m multibyte_composition.compose_four_byte_recovery --campaigns 250 --trace-counts 1,2,5,10,20,50,100,200,500,1000,2000
```

## Outputs

Under `results/multibyte_composition/`:

- `poi_scan/`: the selected positions;
- `byte_attackers/`: per byte, the posteriors, model and training history;
- `byte_attackers/multibyte_composition_summary.csv`: the table rows (`full_successes` out of `campaigns` per `trace_count`, with rank statistics). Bytes 0, 1, 5 and 15 are selected.
