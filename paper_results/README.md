# paper_results

Rebuilds every experimental table and figure of the paper (Tables 7–9, Figures 1–2, and two tables quoted from the full version) and checks each value against the printed one. [Guide 01](../docs/guides/pdf/01_paper_results.pdf) walks through the same steps with recorded terminal output, and the [reproduction checklist](../docs/REPRODUCTION_CHECKLIST.md) maps each table and figure to its experiment.

## Levels

Each level recomputes more of the inputs from public data. All levels run on CPU.

| Level | Command | Recomputes | Needs | Time |
|---|---|---|---|---|
| A | `python -m paper_results.reproduce archived` | nothing; checks the archived counts | nothing | 1 min |
| B | `python -m paper_results.reproduce tches20` | Tables 7–9 and Figure 2 (TCHES20 rows and models) | 0.5 GB download, 8 GB RAM | 8–15 min |
| C | `python -m paper_results.reproduce estranet` | Transformer rows of Tables 7–8, BI line of Figure 1 | 0.3 GB checkpoints + 4.4 GB traces, 8 GB RAM | 20–70 min |
| D | `python -m paper_results.reproduce fullversion` | the two full-version tables | 4.4 GB download, 11 GB free disk | 20 min + download |

`python -m paper_results.reproduce report` rebuilds the tables and figures from whatever has been recomputed so far.

## Setup

With Docker, follow the Quick start of the [main README](../README.md). Without Docker (Python 3.11):

```bash
pip install -r requirements.txt                                                          # Level A
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt  # Levels B–D
```

## Reading a check

Each result ends with one line:

```text
[PASS] Table 7 (finite-suite certificates): 80 values match the paper, 0 differ
```

A value matches when it rounds to the printed number; a `[DIFF]` line lists any value that does not. Levels B and C also report how many recomputed counts equal the archived ones. Tables (`.csv`, `.md`) and figures (`.pdf`, `.png`) are written to `results/paper_results/`.

## Level A: archived counts

```bash
python -m paper_results.reproduce archived
```

Runs the Gaussian oracle of `bayes_oracle/`, then every builder on the archived counts. This exercises the full certificate arithmetic: KL-binomial endpoints, union bounds, margin ratio, radius calibration and conversion to bits.

## Level B: TCHES20 traces and models

```bash
paper_results/fetch_tches20.sh
python -m paper_results.reproduce tches20
```

The run evaluates the 80 pretrained models of each TCHES20 benchmark on the attack traces, computes the relaxed-margin counts of Table 9, profiles the six public reference models of Figure 2, and evaluates EPI and key recovery of the selected models. All 480 success counts equal the archived counts.

Options: `--quick` evaluates only the six selected models (about 2 min), `--datasets` restricts the benchmarks, and an interrupted run resumes where it stopped.

## Level C: EstraNet checkpoints

| Input | Location | How to get it |
|---|---|---|
| EstraNet sources | `external/EstraNet` | `paper_results/fetch_estranet.sh` |
| Checkpoints (308 MB) | `checkpoints/` | `python -m paper_results.fetch_checkpoints` (release file `bi_certification_checkpoints.tar.gz`, SHA-256 `8af739b7295a8ee7d92fef62976e9f8c3d5efac4f10e14fd796db8eea11e89a3`) |
| ASCAD random key (420 MB) | `datasets/raw/ascad_random_key/ASCAD.h5` | `python -m paper_results.fetch_ascad_random_key` |
| CHES-CTF-2025 (4 GB) | `datasets/raw/ches_ctf_2025/CHES_Challenge.h5` | download from https://pace-tl.gitbook.io/ches-challenge-2025 (SHA-256 `132ae2e9a8213c983bf3b63449e9572d5d71d3b376a75b236415d4a728b9379f`) |

```bash
python -m paper_results.reproduce estranet                         # Table 7 and 8 suites
python -m paper_results.reproduce estranet --suites ches_windows   # BI line of Figure 1 (12 GB RAM)
```

Each suite evaluates 11 checkpoints on 100,000 attack traces. The suite maxima (ASCAD random key: checkpoint 80, 1,693 correct; CHES-CTF-2025: checkpoint 5, 429 correct), their EPI and key recovery equal the paper's values.

The Transformer and ASCAD random-key reference models of Figure 2:

```bash
python -m bi_scope_certificates.prepare_aesrd_estranet_h5
for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
  python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
    --output-dir results/paper_results/recomputed/selected_centers/$ds
done
python -m paper_results.reproduce report
```

## Level D: full-version tables

```bash
python -m paper_results.reproduce fullversion
```

- **PI/EPI under temperature rescaling** (`epi_comparison/`): raw PI changes by tens of bits across nine temperatures, while EPI stays within `1e-8` bits and no predicted label changes.
- **Four-byte key recovery on raw ASCAD traces** (`multibyte_composition/`): the check requires median four-byte rank 0 from 500 traces on, and success at 500, 1,000 and 2,000 traces at least as high as printed.

## Reference runs

| Machine | Level A | Level B | Level C (both suites) | Level D |
|---|---|---|---|---|
| Linux workstation, Docker, 24 cores | 21 s | 8 min | 20 min | 12 min |
| Cluster node, Python venv, 16 cores | | 14 min | 69 min | |
| Mac (Apple M3 Pro), Docker | 20 s | 11–14 min | 57 min | 26 min |

On the workstation, the Figure 1 window suites take 34 min and the twelve Figure 2 models 18 min. Every run passes all checks.

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | driver for the levels and `report` |
| `table7_bi_suite.py`, `table8_key_recovery.py`, `table9_biloc_full_suite.py` | Tables 7, 8 and 9 |
| `figure1_dimension_sweep.py`, `figure2_biloc_selected_centers.py` | Figures 1 and 2 |
| `fullversion_tables.py` | the two full-version tables |
| `evaluate_selected_models.py`, `evaluate_estranet_suites.py`, `matched_metrics.py` | EPI and key recovery; the EstraNet suites |
| `fetch_tches20.sh`, `fetch_estranet.sh`, `fetch_ascad_random_key.py`, `fetch_checkpoints.py` | downloads at fixed versions, with checksums |
| `archived/` | counts from the campaigns behind the paper (below) |
| `expected/` | the printed values |

## Output files

| File | Content |
|---|---|
| `table7_bi_suite.csv` | `p_obs` (best success rate in the suite), its KL-binomial interval `[p_minus, p_plus]` at `delta = 1e-6` with the union bound over `M` members, `BI_*` in bits, and `R_margin = (p_plus - p_obs) / (p_obs - 1/256)` |
| `table8_key_recovery.csv` | share of 100 shuffled trace orders in which the key byte ranks first |
| `table9_biloc_full_suite.csv` | per radius: largest relaxed-margin count over the 80 models, its rate, the bound in bits, and the model that attains it |
| `figure1_dimension_sweep.csv`, `figure2_biloc_selected_centers.csv` | the plotted values |

## Archived inputs

`archived/` holds integer counts only: no traces and no model outputs.

| File | Content | Produced by |
|---|---|---|
| `suite_success_counts.csv` | success count of every suite member (6 × 80 TCHES20 models, 2 × 11 checkpoints) | `bi_suite_8datasets/` evaluators |
| `ches_window_success_counts.csv` | the same for the CHES-CTF-2025 window suites (100 to 7,000 samples) | `bi_suite_8datasets/evaluate_estranet_bi_suite.py` |
| `full_suite_relaxed_margin_counts.csv` | relaxed-margin counts of every TCHES20 model at every radius | `bi_scope_certificates/compute_full_suite_bi_loc.py` |
| `selected_center_*.csv` | counts and score variation of the twelve Figure 2 reference models | `bi_scope_certificates/` |
| `selected_model_epi.csv`, `selected_model_key_recovery.csv` | EPI and key recovery of the selected models | `paper_results/evaluate_*.py` |
| `ches_dimension_sweep_nonbi_estimators.csv` | GKOV MI, MLP-PI and ePI/PI values of Figure 1 | `dimension_stability/` |
| `epi_temperature_stress.csv`, `four_byte_composition_1000_campaigns.csv` | the two full-version tables | `epi_comparison/`, `multibyte_composition/` |
