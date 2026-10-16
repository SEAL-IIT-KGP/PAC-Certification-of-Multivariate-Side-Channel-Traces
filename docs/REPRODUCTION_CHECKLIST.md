# Reproduction checklist

Each numerical table and figure of the paper, the code that reproduces it, and the `[PASS]` line its check prints. `python -m paper_results.reproduce <level>` runs all steps of a level at once; the steps below run the same code one file at a time.

## Overview

Level A (`reproduce archived`, no download) rebuilds and checks every result below. The last column says which level recomputes it from the traces and models.

| Paper result | What it shows | Values checked | Recomputed from traces by |
|---|---|---|---|
| [Table 7](#table-7-finite-suite-certificates-and-epi) | Certified leakage bound (bits) for each of the eight benchmarks, with EPI next to it | 80 | Level B (`reproduce tches20`): the six TCHES20 rows<br>Level C (`reproduce estranet`): the ASCAD random key and CHES-CTF-2025 rows |
| [Table 8](#table-8-key-byte-recovery) | How often the best model recovers the key byte with 100 to 2,000 traces | 28 | Level B (`reproduce tches20`): the six TCHES20 rows<br>Level C (`reproduce estranet`): the ASCAD random key row |
| [Table 9](#table-9-full-suite-local-family-certificate) | Certified bound that also covers attackers close to the 80 models, at five radii | 81 | Level B (`reproduce tches20`): all 20 rows |
| [Figure 1](#figure-1-dimension-sweep-on-ches-ctf-2025) | BI stays flat as traces get longer, while the MI and PI estimates drift (BI line at `delta = 0.05`, as in the paper; additional figure at `delta = 1e-6`) | 10 + 2 | Level C (`reproduce estranet`, then `--suites ches_windows`): the BI line<br>Separate long run (step 2, about 18 h): the MI and PI lines |
| [Figure 2](#figure-2-selected-center-local-family-curves) | Certified bound of twelve chosen models as the radius grows, with the chosen radius marked | 12 | Level B (`reproduce tches20`): the six public MLP and CNN models<br>Level C (step 2): all twelve models |
| [Four-byte key recovery](#four-byte-key-recovery-section-53-full-version) (Section 5.3) | Recovering four key bytes together from raw ASCAD traces | 6 (1 in Level A) | Level D (`reproduce fullversion`) |
| [Temperature sweep](#pi-and-epi-under-temperature-rescaling-section-54-full-version) (Section 5.4) | Rescaling a model's confidence changes PI a lot but leaves EPI and BI unchanged | 18 | Level D (`reproduce fullversion`) |
| [Gaussian oracle](#gaussian-oracle-section-51-full-version) (Section 5.1) | Synthetic test where the best possible attacker is known; the trained models nearly match it | 2 | Level A (`reproduce archived`): runs in full, no download |

## Before you start

Build the image and define the `artifact` function as in the [Quick start](../README.md#quick-start); without Docker, run the commands without `artifact`. Download what the steps need, once:

| Data | Command | Used by |
|---|---|---|
| TCHES20 code, 480 pretrained models and traces (0.5 GB) | `artifact paper_results/fetch_tches20.sh` | Tables 7–9, Figure 2, temperature sweep |
| EstraNet sources | `artifact paper_results/fetch_estranet.sh` | Tables 7–8, Figures 1–2 |
| EstraNet and Keras checkpoints (0.3 GB) | `artifact python -m paper_results.fetch_checkpoints` | Tables 7–8, Figures 1–2 |
| ASCAD random key (0.4 GB) | `artifact python -m paper_results.fetch_ascad_random_key` | Tables 7–8, Figure 2 |
| CHES-CTF-2025 (4 GB) | save `CHES_Challenge.h5` from https://pace-tl.gitbook.io/ches-challenge-2025 in `datasets/raw/ches_ctf_2025/` | Table 7, Figure 1 |
| Raw ASCAD traces (4.4 GB download) | `artifact python -m multibyte_composition.fetch_raw_ascad` | four-byte recovery |

The last step of every result is its builder. It reads what was recomputed in `results/paper_results/recomputed/` and the archived counts for the rest, so any subset of steps can be run.

## Table 7: finite-suite certificates and EPI

Checks all eight rows (80 values).

1. Correct guesses of the 80 pretrained models on the six TCHES20 benchmarks (Level B): `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py`
   ```bash
   artifact bash -c 'T=datasets/raw/tches20/TCHES20V3_CNN_SCA
     for ds in ascad_desync_0 ascad_desync_50 ascad_desync_100 aes_hd aes_rd dpav4; do
       python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split --dataset $ds --split attack \
         $([ $ds = aes_rd ] && echo --max-eval-traces 12500) --models-dir $T/models/pretrained_models/models \
         --dataset-dir $T/datasets --tches20-src-dir $T/src \
         --output-dir results/paper_results/recomputed/tches20_suite || exit 1
     done'
   ```
2. EPI of the best model of each benchmark (Level B): `paper_results/evaluate_selected_models.py`
   ```bash
   artifact python -m paper_results.evaluate_selected_models
   ```
3. Correct guesses and EPI of the 11 EstraNet checkpoints for ASCAD random key and CHES-CTF-2025 (Level C): `paper_results/evaluate_estranet_suites.py`
   ```bash
   artifact python -m paper_results.evaluate_estranet_suites --suites ascad_random_key ches_ctf_2025
   ```
4. Build the table and check it: `paper_results/table7_bi_suite.py`
   ```bash
   artifact python -m paper_results.table7_bi_suite
   ```

Expected: `[PASS] Table 7 (finite-suite certificates): 80 values match the paper, 0 differ`

## Table 8: key-byte recovery

Checks the success rate at 100, 500, 1,000 and 2,000 traces for the seven benchmarks (28 values).

1. Key-byte recovery with the best model of each TCHES20 benchmark over 100 trace orders (Level B): `paper_results/evaluate_selected_models.py`
   ```bash
   artifact python -m paper_results.evaluate_selected_models
   ```
2. The same for the best ASCAD random-key checkpoint (Level C): `paper_results/evaluate_estranet_suites.py`
   ```bash
   artifact python -m paper_results.evaluate_estranet_suites --suites ascad_random_key
   ```
3. Build the table and check it: `paper_results/table8_key_recovery.py`
   ```bash
   artifact python -m paper_results.table8_key_recovery
   ```

Expected: `[PASS] Table 8 (key-byte recovery success): 28 values match the paper, 0 differ`

## Table 9: full-suite local-family certificate

Checks the four benchmarks at the five printed radii and the ASCAD d0 bound at radius 0 quoted in Section 5.5 (81 values).

1. Suite bound of each benchmark, read by step 2: `paper_results/table7_bi_suite.py`
   ```bash
   artifact python -m paper_results.table7_bi_suite
   ```
2. Correct and near-correct guesses of the 80 models at nine radii (Level B): `bi_scope_certificates/compute_full_suite_bi_loc.py`
   ```bash
   artifact bash -c 'for ds in aes_rd ascad_desync_0 ascad_desync_50 ascad_desync_100; do
       python -m bi_scope_certificates.compute_full_suite_bi_loc --dataset $ds \
         --suite-csv results/paper_results/bi_suite_dataset_summary.csv \
         --output-dir results/paper_results/recomputed/tches20_full_suite_loc || exit 1
     done'
   ```
3. Build the table and check it: `paper_results/table9_biloc_full_suite.py`
   ```bash
   artifact python -m paper_results.table9_biloc_full_suite
   ```

Expected: `[PASS] Table 9 (full-suite local-family certificates): 81 values match the paper, 0 differ`

## Figure 1: dimension sweep on CHES-CTF-2025

Checks the six plotted BI values, the range quoted in Section 5.4, and the signs of BI and GKOV MI (10 values). The additional figure at `delta = 1e-6`, the confidence level of Table 7, checks the Table 7 value at 3,000 samples (2 values).

1. BI line: correct guesses of the 11 EstraNet checkpoints at each trace length from 100 to 7,000 samples (Level C, 34 min, 12 GB of memory): `paper_results/evaluate_estranet_suites.py`
   ```bash
   artifact python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 ches_windows
   ```
2. Optional, about 18 hours: the MI and PI lines, 54 runs (6 trace lengths × 3 seeds × 3 estimators), listed and then run three at a time: `dimension_stability/ches_nonbi_metric_cell.py`
   ```bash
   artifact python -m dimension_stability.ches_nonbi_metric_cell --write-manifest-only \
     --dimensions 100,200,500,1000,3000,7000 --seeds 0,1,2 --metrics GKOV_MI,MLP_PI,ePI_or_PI \
     --manifest results/paper_results/recomputed/ches_nonbi/manifest.csv \
     --output-dir results/paper_results/recomputed/ches_nonbi
   ```
   ```bash
   artifact bash -c 'D=results/paper_results/recomputed/ches_nonbi
     seq 53 -1 0 | xargs -P 3 -I{} python -m dimension_stability.ches_nonbi_metric_cell \
       --task-id {} --manifest $D/manifest.csv --output-dir $D'
   ```
3. Draw both figures and check them; without step 2 they use the archived MI and PI values: `paper_results/figure1_dimension_sweep.py`
   ```bash
   artifact python -m paper_results.figure1_dimension_sweep
   ```

Expected:
- `[PASS] Figure 1 (dimension sweep): 10 values match the paper, 0 differ`
- `[PASS] Figure 1 at delta = 1e-6 (additional result, confidence level of Table 7): 2 values match the paper, 0 differ`

## Figure 2: selected-center local-family curves

Checks the calibrated radius, the values in the caption and in Section 5.5, and that all twelve certificates are non-vacuous (12 values).

1. Correct and near-correct guesses of the six public models at each radius; 20% of the attack traces choose the radius and 80% give the bound (Level B): `bi_scope_certificates/profile_epsilon_calibration.py`
   ```bash
   artifact bash -c 'for ds in ascad_desync_50 ascad_desync_100 aes_rd; do
       python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds --public-only \
         --output-dir results/paper_results/recomputed/selected_centers/$ds || exit 1
     done'
   ```
2. The same for all twelve models, after preparing the AES-RD input of the Transformer (Level C, 18 min): `bi_scope_certificates/prepare_aesrd_estranet_h5.py`, `bi_scope_certificates/profile_epsilon_calibration.py`
   ```bash
   artifact python -m bi_scope_certificates.prepare_aesrd_estranet_h5
   ```
   ```bash
   artifact bash -c 'for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
       python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
         --output-dir results/paper_results/recomputed/selected_centers/$ds || exit 1
     done'
   ```
3. Choose the radius, draw the figure and check it: `paper_results/figure2_biloc_selected_centers.py`
   ```bash
   artifact python -m paper_results.figure2_biloc_selected_centers
   ```

Expected: `[PASS] Figure 2 (selected-center local-family certificates): 12 values match the paper, 0 differ`

## Four-byte key recovery (Section 5.3, full version)

Checks the median four-byte rank and the success rate at 500, 1,000 and 2,000 traces (6 values in Level D; Level A checks 1).

1. Download the raw ASCAD traces ([Before you start](#before-you-start)): `multibyte_composition/fetch_raw_ascad.py`
2. Find the trace points that leak each key byte: `multibyte_composition/scan_raw_ascad_pois.py`
   ```bash
   artifact python -m multibyte_composition.scan_raw_ascad_pois \
     --output-dir results/paper_results/recomputed/multibyte_composition/poi_scan
   ```
3. Train one attacker for each of eight key bytes (0, 1, 2, 3, 4, 5, 8 and 15): `multibyte_composition/train_raw_ascad_byte_attacker.py`
   ```bash
   artifact bash -c 'R=results/paper_results/recomputed/multibyte_composition
     for b in 0 1 2 3 4 5 8 15; do
       python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte $b \
         --scan-topk $R/poi_scan/raw_ascad_poi_scan_topk.csv --output-dir $R/byte_attackers || exit 1
     done'
   ```
4. Combine the byte attackers to recover four key bytes over 250 campaigns: `multibyte_composition/compose_four_byte_recovery.py`
   ```bash
   artifact python -m multibyte_composition.compose_four_byte_recovery \
     --input-dir results/paper_results/recomputed/multibyte_composition/byte_attackers \
     --campaigns 250 --trace-counts 1,2,5,10,20,50,100,200,500,1000,2000
   ```
5. Build the table and check it (this also builds the temperature table): `paper_results/fullversion_tables.py`
   ```bash
   artifact python -m paper_results.fullversion_tables
   ```

Expected: `[PASS] Four-byte composition (full version): 6 values match the paper, 0 differ`

## PI and EPI under temperature rescaling (Section 5.4, full version)

Checks the raw PI variation, the EPI variation and the predicted labels at every temperature for the six TCHES20 benchmarks (18 values).

1. PI, EPI and predicted key-byte values of one model per benchmark at temperatures from 0.05 to 20 (Level D): `epi_comparison/epi_head_to_head.py`
   ```bash
   artifact bash -c 'T=datasets/raw/tches20/TCHES20V3_CNN_SCA
     for ds in ascad_desync_0 ascad_desync_50 ascad_desync_100 aes_hd aes_rd dpav4; do
       python -m epi_comparison.epi_head_to_head --dataset $ds --dataset-dir $T/datasets \
         --tches20-src-dir $T/src --models-dir $T/models \
         --output-dir results/paper_results/recomputed/epi_comparison/main || exit 1
     done'
   ```
2. Build the table and check it (this also builds the four-byte table): `paper_results/fullversion_tables.py`
   ```bash
   artifact python -m paper_results.fullversion_tables
   ```

Expected: `[PASS] PI/EPI temperature sweep (full version): 18 values match the paper, 0 differ`

## Gaussian oracle (Section 5.1, full version)

Checks the success rate of the Bayes rule and of the learned Gaussian template on synthetic traces with a known posterior (2 values).

1. Run the synthetic test (three seeds): `bayes_oracle/gaussian_oracle_suite.py`
   ```bash
   artifact python -m bayes_oracle.gaussian_oracle_suite --output-dir results/paper_results/gaussian_oracle
   ```
2. Check it; Level A runs step 1 again and then checks every result: `paper_results/reproduce.py`
   ```bash
   artifact python -m paper_results.reproduce archived
   ```

Expected: `[PASS] Synthetic Gaussian oracle (full version, Bayes-reference table): 2 values match the paper, 0 differ`
