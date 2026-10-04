# multibyte_composition

**What this folder does.** It recovers four key bytes of the fixed-key ASCAD capture from the full 100,000-sample raw traces and measures exact four-byte recovery. This is the full version's divide-and-conquer table (`tab:raw-ascad-four-byte-composition`), quoted in Section 5.1 and checked by Level D of `paper_results/`.

## Files, in the order they run

| Step | File | What it does |
|---|---|---|
| 1 | `fetch_raw_ascad.py` | downloads the ANSSI archive `ASCAD_data.zip` (4.4 GB), verifies its SHA-256, extracts the raw-trace member `ATMega8515_raw_traces.h5` (6.0 GB) and verifies that too. An existing file is only verified. `--zip PATH` reuses an archive; `--keep-zip` keeps it. |
| 2 | `scan_raw_ascad_pois.py` | correlates every sample of the first 20,000 traces with three leakage labels per byte (Hamming weight of the unmasked S-box output, of the masked S-box output, and of the mask) and keeps the 20 best positions per label and byte (about 7 minutes on 8 cores) |
| 3 | `train_raw_ascad_byte_attacker.py --target-byte B` | builds a window of 33 samples around the top 8 positions of each label, standardizes it, and trains a two-layer MLP (512 units, dropout 0.1, AdamW, at most 45 epochs with early stopping) on 40,000 traces, with 10,000 validation and 10,000 attack traces (seed 20260520). It stores the attack-split posteriors. The paper trains bytes 0, 1, 2, 3, 4, 5, 8 and 15. |
| 4 | `compose_four_byte_recovery.py` | selects the four bytes with the best validation key rank, then, for each number of traces, draws campaigns of shuffled attack traces, accumulates the per-byte log-probabilities, and counts the campaigns in which the correct four-byte key ranks first under exact enumeration |

## Before you run

- `requirements-ml.txt` (PyTorch; a GPU is optional).
- 11 GB of free disk while the archive is extracted, 6 GB afterwards.

## Run

```bash
python -m multibyte_composition.fetch_raw_ascad
python -m multibyte_composition.scan_raw_ascad_pois
for b in 0 1 2 3 4 5 8 15; do python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte $b; done
python -m multibyte_composition.compose_four_byte_recovery --campaigns 250 --trace-counts 1,2,5,10,20,50,100,200,500,1000,2000
```

## Outputs

Under `results/multibyte_composition/`: `poi_scan/` (the positions), `byte_attackers/` (per byte: posteriors, model, training history, summary), and `byte_attackers/multibyte_composition_summary.csv` with the table rows (`full_successes` out of `campaigns` per `trace_count`, and the rank statistics). Bytes 0, 1, 5 and 15 are selected.

## Where the printed table comes from

The table in the full version (40/250, 94/250, 191/250, 206/250, 221/250 successes at 100 to 2,000 traces) was produced on 2026-05-20 by a composition run that read byte posteriors from a training run which was overwritten a few minutes later by the final training run of the same day. The archived posteriors of that final run, evaluated with the same composition script, give higher success (203/1000 at 100 traces, 445/1000 at 200, 918/1000 at 500, and 1,000/1,000 from 1,000 traces on). `paper_results/fullversion_tables.py` compares a rerun against both sets of numbers. The conclusion drawn in the paper, that a four-byte key is recovered from a few hundred raw traces, holds for both.
