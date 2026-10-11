---
title: "multibyte_composition: recovering four key bytes from raw traces"
---

# What this folder shows

Section 5.1 quotes a result from the long version: using the full raw ASCAD traces (100,000 points each, instead of the usual 700-point window), four key bytes are recovered exactly from a few hundred traces. Four scripts, run in order:

1. `fetch_raw_ascad.py` downloads the ANSSI archive (4.4 GB), checks it, unpacks the 6 GB raw trace file, and checks that too. If the file is already there, it is only checked.
2. `scan_raw_ascad_pois.py` finds the points in the trace that relate to each key byte.
3. `train_raw_ascad_byte_attacker.py --target-byte B` trains a small network for one key byte on those points and saves its guesses on the attack traces. The paper trains bytes 0, 1, 2, 3, 4, 5, 8 and 15.
4. `compose_four_byte_recovery.py` picks the four best bytes and counts, for each number of traces, how often all four are recovered at once.

Level D checks the result.

# Before you start

The Docker image (or the Python environment with `requirements-ml.txt`), 11 GB of free disk while unpacking, and the 4.4 GB download.

# Run

```bash
python -m multibyte_composition.fetch_raw_ascad
```

```{.text transcript="transcripts/06_multibyte_composition/fetch.txt"}
```

```bash
python -m multibyte_composition.scan_raw_ascad_pois
```

```{.text transcript="transcripts/06_multibyte_composition/scan.txt"}
```

```bash
python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte 0
```

```{.text transcript="transcripts/06_multibyte_composition/train_byte0.txt"}
```

```bash
for b in 1 2 3 4 5 8 15; do python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte $b; done
```

```{.text transcript="transcripts/06_multibyte_composition/train_rest.txt"}
```

```bash
python -m multibyte_composition.compose_four_byte_recovery --campaigns 250 \
  --trace-counts 1,2,5,10,20,50,100,200,500,1000,2000
```

```{.text transcript="transcripts/06_multibyte_composition/compose.txt"}
```

```{.text transcript="transcripts/06_multibyte_composition/summary.txt"}
```

# What to look for

- Each training run prints how well the network does on held-out traces. The composition step picks the four best bytes: 0, 1, 5 and 15.
- In the summary, `full_successes` is the number of tries (out of 250) in which all four bytes were right with that many traces. `median_rank` should be 0 from 500 traces on.
