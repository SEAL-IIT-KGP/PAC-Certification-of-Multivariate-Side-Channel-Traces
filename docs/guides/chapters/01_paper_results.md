---
title: "paper_results: rebuilding the tables and figures"
---

# What this folder shows

This folder rebuilds every table and figure of the paper and compares each number with the printed one.

| Result | What it contains | Script |
|----------|------------------------------------------------|------------------------|
| Table 7 | BI for a suite of models on eight sets of traces | `table7_bi_suite.py` |
| Table 8 | how often the best model of each suite finds the key byte | `table8_key_recovery.py` |
| Table 9 | BI for a suite of 80 models, widened to nearby models | `table9_biloc_full_suite.py` |
| Figure 1 | BI stays stable while other information measures drift, as traces get longer | `figure1_dimension_sweep.py` |
| Figure 2 | how far a model can be changed before its BI changes, and the chosen safety margin | `figure2_biloc_selected_centers.py` |
| full-version tables | two results quoted from the long version of the paper | `fullversion_tables.py` |

`archived/` holds the saved counts from the runs behind the paper. `expected/` holds the printed numbers. `reproduce.py` runs the levels.

# Before you start

| Level | You need |
|----|--------------------------------------------------------------|
| A | nothing |
| B | the TCHES20 download (`fetch_tches20.sh`) |
| C | the Transformer source code, the model archive, `ASCAD.h5`, and `CHES_Challenge.h5` |
| D | the TCHES20 download; the raw ASCAD traces are fetched by the level itself |

# Level A: check the saved counts (half a minute, no data)

```bash
python -m paper_results.reproduce archived
```

```{.text transcript="transcripts/01_paper_results/levelA.txt"}
```

To see only the check lines:

```bash
python -m paper_results.reproduce archived --skip-oracle 2>/dev/null | grep -E '^\[(PASS|DIFF)\]'
```

```{.text transcript="transcripts/01_paper_results/levelA_checks.txt"}
```

The tables and figures are written to `results/paper_results/`:

```{.text transcript="transcripts/01_paper_results/levelA_files.txt"}
```

```{.text transcript="transcripts/01_paper_results/table7.txt"}
```

```{.text transcript="transcripts/01_paper_results/table8.txt"}
```

```{.text transcript="transcripts/01_paper_results/table9.txt"}
```

![Figure 1, as rebuilt by the code.](images/figure1_dimension_sweep.png){width=85%}

![Figure 2, as rebuilt by the code.](images/figure2_biloc_selected_centers.png){width=85%}

**What to look for.** Eight `[PASS]` lines, and `matches the paper` for every result. In the Table 7 file, `p_obs` is the success rate of the best model in the suite, `p_plus` is the guaranteed upper limit on that rate, and `BI_plus` is the same limit in bits. The last column says whether a row used saved or recomputed counts.

# Level B: recompute from the TCHES20 traces and models

Download the traces and the 480 trained models (about a minute):

```bash
paper_results/fetch_tches20.sh
```

```{.text transcript="transcripts/01_paper_results/fetch_tches20.txt"}
```

Run the level. It tests all 80 models of each trace set, counts their correct guesses, and rebuilds the tables from those counts:

```bash
python -m paper_results.reproduce tches20
```

```{.text transcript="transcripts/01_paper_results/levelB.txt"}
```

```{.text transcript="transcripts/01_paper_results/levelB_counts.txt"}
```

**What to look for.** `80/80 recomputed per-model counts equal the archived counts` for each of the six trace sets. A count is the number of traces where the model's top guess is right, so it comes out the same on any machine.

Shortcuts: `--quick` tests only the six best models (two minutes). `--datasets` limits the run to some trace sets. A stopped run continues where it left off.

# Level C: recompute with the Transformer models

Get the Transformer source code:

```bash
paper_results/fetch_estranet.sh
```

```{.text transcript="transcripts/01_paper_results/fetch_estranet.txt"}
```

Download, check and unpack the provided models (308 MB):

```bash
python -m paper_results.fetch_checkpoints
ls checkpoints/estranet checkpoints/keras_ascad_random_key
```

```{.text transcript="transcripts/01_paper_results/checkpoints.txt"}
```

Download the ASCAD random-key traces:

```bash
python -m paper_results.fetch_ascad_random_key
```

```{.text transcript="transcripts/01_paper_results/fetch_ascad_random.txt"}
```

Download `CHES_Challenge.h5` by hand from https://pace-tl.gitbook.io/ches-challenge-2025. Put it in the folder `datasets/raw/ches_ctf_2025/` and check it:

```{.text transcript="transcripts/01_paper_results/ches_file.txt"}
```

Run the two Table 7 suites. Each one tests eleven saved models on 100,000 traces:

```bash
python -m paper_results.reproduce estranet --suites ascad_random_key ches_ctf_2025
```

```{.text transcript="transcripts/01_paper_results/levelC.txt"}
```

Optional: the five extra suites behind the BI line of Figure 1 (models trained on shorter and longer traces), and the twelve models of Figure 2:

```bash
python -m paper_results.reproduce estranet --suites ches_windows
```

```{.text transcript="transcripts/01_paper_results/levelC_windows.txt"}
```

```bash
python -m bi_scope_certificates.prepare_aesrd_estranet_h5
for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
  python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
    --output-dir results/paper_results/recomputed/selected_centers/$ds
done
python -m paper_results.reproduce report
```

```{.text transcript="transcripts/01_paper_results/levelC_figure2.txt"}
```

**What to look for.** The best model of each suite is the same as in the paper (ASCAD random key: model 80 with 1,693 correct guesses out of 100,000; CHES-CTF-2025: model 5 with 429), and every check ends with `[PASS]`. For Figure 2, the output prints the chosen safety margin, `0.0003`, as in the paper.

# Level D: the two results quoted from the long version

```bash
python -m paper_results.reproduce fullversion
```

```{.text transcript="transcripts/01_paper_results/levelD.txt"}
```

**What to look for.** First, a comparison of two information measures on six trace sets. Then the raw ASCAD traces are downloaded and checked, points of interest are found, eight small models are trained (one per key byte), and four of them are combined to recover four key bytes at once. The check ends with `[PASS]` for both results. The four-byte check tests that the four key bytes are recovered from 500 traces on.

`python -m paper_results.reproduce report` rebuilds the tables from whatever has been recomputed so far.
