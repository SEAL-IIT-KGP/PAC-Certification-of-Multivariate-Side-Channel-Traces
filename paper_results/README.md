# Reproducing the Paper's Tables and Figures

This directory rebuilds every experimental table and figure of the ASIACRYPT 2026
paper and compares each regenerated value with the value printed in the paper.
Numbering follows the camera-ready version (Tables 7-9, Figures 1-2).

| Result | Content | Builder |
|---|---|---|
| Table 7 | Finite-suite certificates `BI_suite` on eight benchmarks, with EPI | `table7_bi_suite.py` |
| Table 8 | Key-byte recovery success of the selected suite members | `table8_key_recovery.py` |
| Table 9 | Full-suite local-family certificate `BI_loc,+(eps)` over M = 80 models | `table9_biloc_full_suite.py` |
| Figure 1 | CHES-CTF-2025 dimension sweep: `BI_suite,+` and density-style estimators | `figure1_dimension_sweep.py` |
| Figure 2 | Selected-center local-family curves and the calibrated radius | `figure2_biloc_selected_centers.py` |

There are three levels. Each level replaces more of the archived inputs with
values recomputed from the traces.

| Level | Command | Starts from | Needs | Time |
|---|---|---|---|---|
| A | `python -m paper_results.reproduce archived` | archived per-model counts in `archived/` | nothing (no downloads) | about 1 minute |
| B | `python -m paper_results.reproduce tches20` | public TCHES20 traces and its 480 pretrained models | 0.5 GB download, 2.3 GB on disk, 8 GB RAM, CPU | 8 to 15 minutes on 16 to 24 cores |
| C | `python -m paper_results.reproduce estranet` | released EstraNet checkpoints, ASCAD random-key and CHES-CTF-2025 traces | checkpoints (308 MB) and two datasets (0.4 GB and 4 GB), 8 GB RAM (16 GB for the 7,000-sample window), CPU | 20 to 70 minutes on 24 to 16 cores for the two Table 7 suites |
| D | `python -m paper_results.reproduce fullversion` | public TCHES20 models and the raw ASCAD traces | 4.4 GB download (the ANSSI `ASCAD_data.zip`), 11 GB on disk while extracting, PyTorch, CPU | about 20 minutes on 24 cores, plus the download (6 minutes at 15 MB/s) |

Level B covers the six TCHES20 rows of Table 7 (counts, certificates, EPI), the
six TCHES20 rows of Table 8, all of Table 9, and the six MLP/CNN reference
models of Figure 2 that are public TCHES20 models. Level C covers the two
EstraNet rows of Table 7, the ASCAD random-key row of Table 8, and the
`BI_suite,+` line of Figure 1.

## Setup

With Docker (from the repository root):

```bash
docker build -t bi-certification .
mkdir -p results datasets/raw
```

Without Docker (Python 3.11):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
```

Level A needs only `requirements.txt`.

## Level A: archived counts

```bash
docker run --rm -v "$PWD/results:/artifact/results" bi-certification
# or, without Docker:
python -m paper_results.reproduce archived
```

This runs the synthetic Gaussian oracle of `bayes_oracle/` (no data needed),
then the five builders. Each builder prints its table and one line per result:

```text
[PASS] Table 7 (finite-suite certificates): 80 values match the paper, 0 differ
[PASS] Table 8 (key-byte recovery success): 28 values match the paper, 0 differ
[PASS] Table 9 (full-suite local-family certificates): 81 values match the paper, 0 differ
[PASS] Figure 1 (dimension sweep): 5 values match the paper, 0 differ
[PASS] Figure 2 (selected-center local-family certificates): 12 values match the paper, 0 differ
```

A value matches when it rounds to the number printed in the paper. A `[DIFF]`
line lists every value that does not. The tables (`.csv` and `.md`) and the
figures (`.pdf` and `.png`) are written to `results/paper_results/`.

The certificates are computed from integer counts, so Level A checks the full
certificate arithmetic: the KL-binomial endpoints, the confidence split
`delta / (2M)` of the suite certificate, the split `delta / (M |grid|)` of the
local-family certificate, the margin ratio, the radius calibration rule, and
the conversion to bits. The EPI column, the key-recovery percentages of
Table 8, and the density-style estimator values of Figure 1 are read from the
archived evaluation summaries at this level; Levels B and C recompute EPI and
key recovery from the model predictions.

## Level B: from the public TCHES20 traces and models

```bash
# Docker
docker run --rm -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
  bi-certification bash -c "paper_results/fetch_tches20.sh && python -m paper_results.reproduce tches20"

# without Docker
paper_results/fetch_tches20.sh
python -m paper_results.reproduce tches20
```

`fetch_tches20.sh` clones https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA
(datasets, loaders, and pretrained models) at a fixed commit into
`datasets/raw/tches20/TCHES20V3_CNN_SCA` and unpacks it. The run then

1. evaluates all 80 pretrained models of each benchmark on the attack traces
   (`bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split --split attack`;
   AES-RD uses the first 12,500 attack traces) and writes the per-model success counts;
2. computes the relaxed-margin counts of the four 80-model suites over the nine
   radii (`bi_scope_certificates.compute_full_suite_bi_loc`);
3. profiles the six public MLP/CNN reference models of Figure 2 with the 20%/80%
   calibration split (`bi_scope_certificates.profile_epsilon_calibration --public-only`);
4. evaluates the selected model of each benchmark for EPI and key recovery
   (`paper_results.evaluate_selected_models`);
5. rebuilds the tables and figures, using recomputed counts wherever they exist.

The builders report how many recomputed per-model counts equal the archived
counts, for example `ascad_desync_0: 80/80 recomputed per-model counts equal
the archived counts`. Success counts are arg-max decisions and are expected to
agree exactly. Relaxed-margin counts compare float32 score margins with a
threshold, so a count can move by a trace on other hardware; the checks
for recomputed relaxed-margin counts therefore allow 0.01 bits.

Reference runs (Linux x86-64, the pinned versions): in the Docker image on a
24-core workstation (image build 2.5 minutes, 3.5 GB; Level A 21 seconds;
fetch and Level B 8 minutes) and in a Python 3.11 virtual environment on a
16-core cluster node (Level B 14 minutes). In both, all 480 success counts
equal the archived counts, 2,877 of the 2,880 (model, radius) relaxed-margin
counts are equal and three differ by one trace, and every check reports
`[PASS]`.

`python -m paper_results.reproduce tches20 --quick` evaluates only the six
selected models (step 4): it reproduces the six success counts behind
`p_obs`, the EPI column, and Table 8 in about two minutes. `--datasets` restricts a
run to some benchmarks, and an interrupted run resumes from the counts already
written under `results/paper_results/recomputed/`.

## Level C: EstraNet checkpoint suites

The ASCAD random-key and CHES-CTF-2025 rows use EstraNet Transformer
checkpoints trained for this paper (`bi_suite_8datasets/estranet_support/README.md`
lists the training settings). Level C needs:

| Input | Default location | How to obtain |
|---|---|---|
| EstraNet sources | `external/EstraNet` | `paper_results/fetch_estranet.sh` |
| Released checkpoints (162 files, 308 MB) | `checkpoints/estranet/<suite>/trans_long-<index>.*`, `checkpoints/keras_ascad_random_key/*.hdf5` | extract `bi_certification_checkpoints.tar.gz` (SHA-256 `8323e217130bd7e7db7a3f191ed9963b5304494cb83bec4f974902f10c296762`) in the repository root |
| ASCAD random-key database, 1,400-sample window (420 MB) | `datasets/raw/ascad_random_key/ASCAD.h5` | `python -m paper_results.fetch_ascad_random_key` |
| CHES-CTF-2025 traces, `CHES_Challenge.h5` (4 GB) | `datasets/raw/ches_ctf_2025/CHES_Challenge.h5` | public dataset of the CHES Challenge 2025, https://pace-tl.gitbook.io/ches-challenge-2025 (SHA-256 `132ae2e9a8213c983bf3b63449e9572d5d71d3b376a75b236415d4a728b9379f`) |

```bash
paper_results/fetch_estranet.sh
python -m paper_results.fetch_ascad_random_key
python -m paper_results.reproduce estranet --suites ascad_random_key ches_ctf_2025
python -m paper_results.reproduce estranet --suites ches_windows   # BI line of Figure 1
```

Each suite evaluates 11 checkpoints on 100,000 attack traces. For the two
Table 7 suites, the checkpoint with the largest attack success is also
evaluated for EPI and key recovery.

Reference runs (CPU only): in the Docker image on a 24-core workstation (11
minutes for the ASCAD random-key suite, 9 minutes for the CHES-CTF-2025 suite,
34 minutes for the five window suites) and in a Python 3.11 virtual
environment on a 16-core cluster node (69 minutes for both Table 7 suites).
The suite maxima (ASCAD random key: checkpoint 80 with 1,693 of 100,000;
CHES-CTF-2025: checkpoint 5 with 429 of 100,000), their EPI, and their
key-recovery percentages equal the paper's values. All per-checkpoint counts
of the CHES-CTF-2025 suites on 3,000 and 7,000 samples equal the archived
ones. For the ASCAD random-key suite and the shorter CHES-CTF-2025 windows,
two or three of the eleven near-chance counts per suite differ by one trace
between machines (up to four traces for the earliest ASCAD random-key
checkpoint), which moves `BI_suite,+` by at most 0.003 bits.

Memory: every run above stays within 8 GB (the default limit of Docker
Desktop) except the 7,000-sample window suite, which peaks at 12 GB.

The four Transformer and the two ASCAD random-key MLP/CNN reference models of
Figure 2 are profiled by `bi_scope_certificates.profile_epsilon_calibration`
(without `--public-only`) from the same checkpoint release:

```bash
python -m bi_scope_certificates.prepare_aesrd_estranet_h5     # AES-RD input of the Transformer model
for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
  python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
    --output-dir results/paper_results/recomputed/selected_centers/$ds
done
python -m paper_results.reproduce report
```

In the reference run (Docker image, 24 cores, 18 minutes), 208 of the 216
relaxed-margin counts of the twelve models equal the archived counts and eight
differ by one trace.

With Docker, mount the checkpoint, dataset, and EstraNet directories next to
the results:

```bash
docker run --rm -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
  -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
  bi-certification bash -c "paper_results/fetch_estranet.sh && python -m paper_results.fetch_ascad_random_key \
    && python -m paper_results.reproduce estranet --suites ascad_random_key ches_ctf_2025"
```

## Level D: results quoted from the full version

Section 5 quotes two results of the full version. Both have their code in this
repository and are checked by `paper_results/fullversion_tables.py`, which the
report runs after the camera-ready builders.

```bash
python -m paper_results.reproduce fullversion
```

- **PI/EPI under temperature rescaling** (`epi_comparison/`, public TCHES20
  models): for each benchmark, the most accurate of the 80 models on a
  class-balanced selection split is frozen, EPI is calibrated on a disjoint
  split, and raw PI and recalibrated EPI are recomputed at nine temperatures
  from 0.05 to 20. The check compares the variation of raw PI (to 0.05 bits for
  a rerun) and requires the EPI variation to stay below 10⁻⁸ bits with every
  predicted label preserved.
- **Four-byte key recovery on raw ASCAD traces** (`multibyte_composition/`,
  4.4 GB archive download, PyTorch): points of interest are scanned on the raw traces,
  one byte attacker is trained for each of eight bytes, and the four best
  bytes are composed into exact four-byte recovery over 250 campaigns. The
  printed table came from byte attackers that a later training run overwrote
  (`multibyte_composition/README.md`), so the check is on the claim rather
  than on the exact counts: median four-byte rank 0 from 500 traces on, and a
  success rate at 500, 1,000 and 2,000 traces at least as high as printed.
  The archived attackers of the final run give 918 of 1,000 at 500 traces and
  1,000 of 1,000 from 1,000 traces on. A CPU rerun of the whole pipeline in the
  Docker image (24 cores, 12 minutes after the download) selected the same four
  bytes and gave 207, 245 and 250 of 250 at 500, 1,000 and 2,000 traces.

## Reading the outputs

- `table7_bi_suite.csv`: `p_obs` is the largest attack success rate in the
  suite, `[p_minus, p_plus]` its two-sided KL-binomial interval at
  `delta = 1e-6` with the union bound over the `M` suite members, `BI_*` the
  same quantities in bits (`log2(256 p)`, floored at 0), and
  `R_margin = (p_plus - p_obs) / (p_obs - 1/256)`. `R_margin < 1` marks a row
  driven by the observed advantage; `R_margin > 1` marks a row where the
  finite-sample margin dominates. `counts_source` says whether the counts were
  archived or recomputed.
- `table8_key_recovery.csv`: percentage of 100 shuffled trace orders in which
  the correct key byte ranks first after the given number of traces.
- `table9_biloc_full_suite.csv`: for each radius, the largest relaxed-margin
  count `S_loc` over the 80 models, its rate, the certified endpoint in bits,
  and the architecture of the model that attains it. All nine radii are listed;
  the paper prints five.
- `figure1_dimension_sweep.csv` and `figure2_biloc_selected_centers.csv`: the
  plotted values. `figure2_calibrated_radius_points.csv` lists the twelve
  certificates at the calibrated radius.

## Archived inputs

`archived/` holds the integer counts produced by the evaluation campaigns
behind the paper. They contain no traces and no model outputs beyond counts.

| File | Content | Produced by |
|---|---|---|
| `suite_success_counts.csv` | attack success count of every suite member (6 x 80 TCHES20 models, 2 x 11 EstraNet checkpoints) | `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py`, `evaluate_estranet_bi_suite.py` |
| `ches_window_success_counts.csv` | the same counts for the CHES-CTF-2025 suites trained on windows of 100 to 7,000 samples | `bi_suite_8datasets/evaluate_estranet_bi_suite.py` |
| `full_suite_relaxed_margin_counts.csv` | relaxed-margin count of every TCHES20 model at every radius (4 benchmarks) | `bi_scope_certificates/compute_full_suite_bi_loc.py` |
| `selected_center_relaxed_margin_counts.csv`, `selected_center_score_variation.csv` | relaxed-margin counts on the calibration and certification splits, and score variation, for the twelve reference models | `bi_scope_certificates/profile_epsilon_calibration.py` |
| `selected_center_full_attack_relaxed_margin_counts.csv` | relaxed-margin counts of the same twelve models on their full attack sets | `bi_scope_certificates/compute_bi_loc_margin_curves.py` |
| `selected_model_epi.csv`, `selected_model_key_recovery.csv` | EPI and key-recovery campaign counts of the eight selected models | `paper_results/evaluate_selected_models.py`, `evaluate_estranet_suites.py` |
| `ches_dimension_sweep_nonbi_estimators.csv` | GKOV MI, MLP-PI, and ePI/PI values of Figure 1, with a `status` per cell | `dimension_stability/ches_nonbi_metric_cell.py`, `real_nonbi_dimcap_baselines.py` |
| `ches_dimension_sweep_estimator_rerun.csv` | per-seed rerun of all 54 estimator cells of Figure 1 after the camera-ready (2026-10-01/02) in two environments (cluster venv and the Docker image), with runtime | `dimension_stability/ches_nonbi_metric_cell.py` |

`expected/` holds the values printed in the paper, as typed from the
camera-ready PDF.

## Notes on scope

- **Figure 1, density-style estimators.** The GKOV MI and MLP-PI curves are
  measured at all six dimensions. In the camera-ready figure, the ePI/PI
  values at 3,000 and 7,000 samples are log-linear extrapolations of the
  measured trend (−0.760 and −0.984 bits), because the template estimator's
  runtime grows with the dimension (about three hours per seed at 1,000
  samples). These two cells were measured after the camera-ready with the
  repository's `dimension_stability/ches_nonbi_metric_cell.py` and the paper's
  settings (three seeds each, 5 and 15 hours per seed on 16 CPU cores):
  −0.900 bits at 3,000 and −1.168 bits at 7,000 samples. The archived data
  file carries the measured means (`status = measured`, per-seed values in
  `ches_dimension_sweep_epi_measurements.csv`) and notes the camera-ready
  values, so the regenerated figure shows the measured points, which lie
  below the printed ones. The paper's statements about the figure (the
  perceived-information estimators drift downward while `BI_suite,+` stays
  flat) are unchanged. One more provenance detail: the camera-ready GKOV MI
  cell at 100 samples was computed with the estimator subsampled to 4,000 of
  the 24,000 training traces, while the cells at 200 samples and above, and
  `ches_nonbi_metric_cell.py`, subsample 8,000. Recomputing the 100-sample cell
  with 8,000 traces gives about −0.49 bits instead of the printed −0.39; the
  archived data file notes this, and the figure's checks do not depend on it.
  Of the three estimators, GKOV MI and ePI/PI recompute to three decimals,
  and a cluster rerun and a Docker rerun of all 54 cells agree with each other
  to the last digit at every seed (`ches_dimension_sweep_estimator_rerun.csv`).
  MLP-PI is a small network trained for seconds per cell; with the same seed it
  gives different values on different machines, and it moves by a few tenths
  of a bit between seeds (occasionally by more) in the original campaign and in
  both reruns, so its line is a three-seed mean of training noise. Every
  MLP-PI cell stays at or below zero, which is what the figure uses it for. A
  rerun's recomputed means replace the archived ones in the regenerated figure.
- **Calibrated radius and the inference platform.** The radius is the first
  grid value that (i) changes the relaxed-margin count of at least 75% of the
  twelve reference models on the calibration split, which gives `1e-4`, and
  (ii) covers the largest score change seen under repeated FP32 inference and
  other batch sizes. On the GPU runs behind the paper, the Transformer scores
  change with the batch size by up to `1.9e-4`, so the radius is `3e-4`. CPU
  inference is deterministic across batch sizes (score change below `1e-6`).
  The builder therefore keeps the archived GPU score-variation profile, adds
  the profile of a recomputation, reports both, and selects the radius that
  covers both.
- **Section 5.3, values at `eps = 1e-4`.** Figure 2 certifies on the 80%
  certification split with the confidence split `delta / (12 |grid|)`. The
  twelve bounds quoted at `eps = 1e-4` in the text use each model's full attack
  set with the split `delta / |grid|`; they are checked from
  `selected_center_full_attack_relaxed_margin_counts.csv`.
- **Table 8.** The paper's table lists seven benchmarks. The builder also
  prints the CHES-CTF-2025 row for completeness.
- **Results quoted from the full version.** The four-byte key recovery on raw
  ASCAD traces and the temperature-rescaling comparison are covered by Level D.
  The remaining appendix diagnostics of the full version have their scripts
  listed in `docs/EXPERIMENT_MAP.md`.
