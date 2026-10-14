# Reproduction checklist

Each numerical table and figure of the camera-ready paper, the code that reproduces it step by step, and the check that compares the result with the printed values. Every check ends with one `[PASS]` line; tick the box when it appears.

Level A rebuilds every result from the archived counts in `paper_results/archived/`. Levels B, C and D recompute those counts from the public traces and models. `python -m paper_results.reproduce <level>` runs all steps of a level at once ([By level](#by-level)); the steps below run the same code one file at a time.

## Overview

Level A (`reproduce archived`, no download) rebuilds and checks every result below. The last column says which level recomputes it from the traces and models.

| Paper result | What it shows | Values checked | Recomputed from traces by |
|---|---|---|---|
| [Table 7](#table-7-finite-suite-certificates-and-epi) | Certified leakage bound (bits) for each of the eight benchmarks, with EPI next to it | 80 | Level B (`reproduce tches20`): the six TCHES20 rows<br>Level C (`reproduce estranet`): the ASCAD random key and CHES-CTF-2025 rows |
| [Table 8](#table-8-key-byte-recovery) | How often the best model recovers the key byte with 100 to 2,000 traces | 28 | Level B (`reproduce tches20`): the six TCHES20 rows<br>Level C (`reproduce estranet`): the ASCAD random key row |
| [Table 9](#table-9-full-suite-local-family-certificate) | Certified bound that also covers attackers close to the 80 models, at five radii | 81 | Level B (`reproduce tches20`): all 20 rows |
| [Figure 1](#figure-1-dimension-sweep-on-ches-ctf-2025) | BI stays flat as traces get longer, while the MI and PI estimates drift | 5 | Level C (`reproduce estranet`, then `--suites ches_windows`): the BI line<br>Separate long run (steps 2–3, about 18 h): the MI and PI lines |
| [Figure 2](#figure-2-selected-center-local-family-curves) | Certified bound of twelve chosen models as the radius grows, with the chosen radius marked | 12 | Level B (`reproduce tches20`): the six public MLP and CNN models<br>Level C (steps 2–3): all twelve models |
| [Four-byte key recovery](#four-byte-key-recovery-section-53-full-version) (Section 5.3) | Recovering four key bytes together from raw ASCAD traces | 6 (1 in Level A) | Level D (`reproduce fullversion`) |
| [Temperature sweep](#pi-and-epi-under-temperature-rescaling-section-54-full-version) (Section 5.4) | Rescaling a model's confidence changes PI a lot but leaves EPI and BI unchanged | 18 | Level D (`reproduce fullversion`) |
| [Gaussian oracle](#gaussian-oracle-section-51-full-version) (Section 5.1) | Synthetic test where the best possible attacker is known; the trained models nearly match it | 2 | Level A (`reproduce archived`): runs in full, no download |

## Before you start

Build the image and define `artifact`, which runs a command in the container with the four data folders shared (as in the [Quick start](../README.md#quick-start)). Without Docker, run the same commands without `artifact`.

```bash
docker build -t bi-certification .
mkdir -p results datasets/raw checkpoints external
artifact() {
  docker run --rm -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
    -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
    bi-certification "$@"
}
```

Download what the steps need, once:

| Data | Command | Used by |
|---|---|---|
| TCHES20 code, 480 pretrained models and traces (0.5 GB) | `artifact paper_results/fetch_tches20.sh` | Tables 7–9, Figure 2, temperature sweep |
| EstraNet sources | `artifact paper_results/fetch_estranet.sh` | Tables 7–8, Figures 1–2 |
| EstraNet and Keras checkpoints (0.3 GB) | `artifact python -m paper_results.fetch_checkpoints` | Tables 7–8, Figures 1–2 |
| ASCAD random key (0.4 GB) | `artifact python -m paper_results.fetch_ascad_random_key` | Tables 7–8, Figure 2 |
| CHES-CTF-2025 (4 GB) | save `CHES_Challenge.h5` from https://pace-tl.gitbook.io/ches-challenge-2025 in `datasets/raw/ches_ctf_2025/` | Table 7, Figure 1 |
| Raw ASCAD traces (4.4 GB download) | `artifact python -m multibyte_composition.fetch_raw_ascad` | four-byte recovery |

Each builder (the last step of every result) reads recomputed results from `results/paper_results/recomputed/` and uses the archived counts for anything not recomputed, so any subset of steps can be run.

## Table 7: finite-suite certificates and EPI

Checks `M`, `n_te`, `p_obs`, `[p-, p+]`, `BI_obs`, `[BI-, BI+]`, `R_margin` and EPI of all eight rows (80 values).

1. **Count how many attack traces each of the 80 pretrained models guesses correctly**, for the six TCHES20 benchmarks (Level B): `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py`
   ```bash
   artifact bash -c 'T=datasets/raw/tches20/TCHES20V3_CNN_SCA
     for ds in ascad_desync_0 ascad_desync_50 ascad_desync_100 aes_hd aes_rd dpav4; do
       python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split --dataset $ds --split attack \
         $([ $ds = aes_rd ] && echo --max-eval-traces 12500) --models-dir $T/models/pretrained_models/models \
         --dataset-dir $T/datasets --tches20-src-dir $T/src \
         --output-dir results/paper_results/recomputed/tches20_suite || exit 1
     done'
   ```
2. **Compute EPI for the best model of each benchmark**, the one with the most correct guesses in step 1 (Level B): `paper_results/evaluate_selected_models.py`
   ```bash
   artifact python -m paper_results.evaluate_selected_models
   ```
3. **Count the correct guesses of the 11 EstraNet checkpoints** on 100,000 attack traces, and compute EPI of the best one, for ASCAD random key and CHES-CTF-2025 (Level C): `paper_results/evaluate_estranet_suites.py`
   ```bash
   artifact python -m paper_results.evaluate_estranet_suites --suites ascad_random_key ches_ctf_2025
   ```
4. **Compute the bounds, build the table and compare it with the paper**: `paper_results/table7_bi_suite.py`
   ```bash
   artifact python -m paper_results.table7_bi_suite
   ```

- [ ] `[PASS] Table 7 (finite-suite certificates): 80 values match the paper, 0 differ`
- [ ] after step 1, step 4 also prints `80/80 recomputed per-model counts equal the archived counts` for each TCHES20 benchmark

## Table 8: key-byte recovery

Checks the success rate at 100, 500, 1,000 and 2,000 traces for the seven benchmarks, dashes included (28 values).

1. **Attack with the best model of each TCHES20 benchmark**, adding up traces in 100 random orders, and count how often the correct key byte comes first (Level B): `paper_results/evaluate_selected_models.py`
   ```bash
   artifact python -m paper_results.evaluate_selected_models
   ```
2. **The same for the best ASCAD random-key checkpoint** (Level C): `paper_results/evaluate_estranet_suites.py`
   ```bash
   artifact python -m paper_results.evaluate_estranet_suites --suites ascad_random_key
   ```
3. **Build the table and compare it with the paper**: `paper_results/table8_key_recovery.py`
   ```bash
   artifact python -m paper_results.table8_key_recovery
   ```

- [ ] `[PASS] Table 8 (key-byte recovery success): 28 values match the paper, 0 differ`

## Table 9: full-suite local-family certificate

Checks `S^loc`, `p^loc`, `BI^loc,+` and the architecture at the five printed radii for AES-RD and ASCAD d0/d50/d100 (80 values), and the 1.709-bit ASCAD d0 bound at radius 0 quoted in Section 5.5 (1 value).

1. **Write the suite bound of each benchmark**, which step 2 reads: `paper_results/table7_bi_suite.py`
   ```bash
   artifact python -m paper_results.table7_bi_suite
   ```
2. **For each of the 80 models and nine radii, count the attack traces it guesses correctly or within the radius of a correct guess** (Level B): `bi_scope_certificates/compute_full_suite_bi_loc.py`
   ```bash
   artifact bash -c 'for ds in aes_rd ascad_desync_0 ascad_desync_50 ascad_desync_100; do
       python -m bi_scope_certificates.compute_full_suite_bi_loc --dataset $ds \
         --suite-csv results/paper_results/bi_suite_dataset_summary.csv \
         --output-dir results/paper_results/recomputed/tches20_full_suite_loc || exit 1
     done'
   ```
3. **Compute the bounds, build the table and compare it with the paper**: `paper_results/table9_biloc_full_suite.py`
   ```bash
   artifact python -m paper_results.table9_biloc_full_suite
   ```

- [ ] `[PASS] Table 9 (full-suite local-family certificates): 81 values match the paper, 0 differ`

## Figure 1: dimension sweep on CHES-CTF-2025

Checks the smallest and largest `BI^suite,+` over the sweep (0.47 and 0.55 bits, Section 5.4), `BI^suite,+` at 3,000 samples (0.521 bits, Table 7), a positive `BI^suite,+` at every dimension, and a negative GKOV MI at 7,000 samples (5 values).

1. **BI line: count the correct guesses of the 11 EstraNet checkpoints for each trace length**, 100 to 7,000 samples (Level C, 34 min, 12 GB of memory): `paper_results/evaluate_estranet_suites.py`
   ```bash
   artifact python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 ches_windows
   ```
2. **MI and PI lines: list the 54 runs** (6 trace lengths × 3 seeds × 3 estimators): `dimension_stability/ches_nonbi_metric_cell.py`
   ```bash
   artifact python -m dimension_stability.ches_nonbi_metric_cell --write-manifest-only \
     --dimensions 100,200,500,1000,3000,7000 --seeds 0,1,2 --metrics GKOV_MI,MLP_PI,ePI_or_PI \
     --manifest results/paper_results/recomputed/ches_nonbi/manifest.csv \
     --output-dir results/paper_results/recomputed/ches_nonbi
   ```
3. **MI and PI lines: do the 54 runs**, three at a time, longest first (about 18 hours in total; each 7,000-sample ePI/PI run takes 12 to 15 hours): `dimension_stability/ches_nonbi_metric_cell.py`
   ```bash
   artifact bash -c 'D=results/paper_results/recomputed/ches_nonbi
     seq 53 -1 0 | xargs -P 3 -I{} python -m dimension_stability.ches_nonbi_metric_cell \
       --task-id {} --manifest $D/manifest.csv --output-dir $D'
   ```
4. **Draw the figure and compare it with the paper**; it uses the archived MI and PI values if steps 2–3 were skipped: `paper_results/figure1_dimension_sweep.py`
   ```bash
   artifact python -m paper_results.figure1_dimension_sweep
   ```

- [ ] `[PASS] Figure 1 (dimension sweep): 5 values match the paper, 0 differ`

## Figure 2: selected-center local-family curves

Checks the calibrated radius `eps_cal = 3e-4` (Section 5.1); the caption values at `eps_cal` (AES-RD MLP, CNN and Transformer 3.07, 5.36 and 6.83 bits; ASCAD range 1.76 to 3.48 bits); the values at `eps = 1e-4` quoted in Section 5.5 (2.85, 5.34 and 6.80 bits; ASCAD range 1.5 to 2.3 bits); and twelve non-vacuous certificates at `eps_cal` (12 values).

1. **For the six public models, count correct and near-correct guesses at each radius** (MLP and CNN on ASCAD d50, ASCAD d100 and AES-RD); 20% of the attack traces choose the radius and 80% give the bound (Level B): `bi_scope_certificates/profile_epsilon_calibration.py`
   ```bash
   artifact bash -c 'for ds in ascad_desync_50 ascad_desync_100 aes_rd; do
       python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds --public-only \
         --output-dir results/paper_results/recomputed/selected_centers/$ds || exit 1
     done'
   ```
2. **Prepare the AES-RD input file for the Transformer model** (Level C): `bi_scope_certificates/prepare_aesrd_estranet_h5.py`
   ```bash
   artifact python -m bi_scope_certificates.prepare_aesrd_estranet_h5
   ```
3. **The same counts for all twelve models**, adding the four Transformers and the MLP and CNN on ASCAD random key (Level C, 18 min with step 2): `bi_scope_certificates/profile_epsilon_calibration.py`
   ```bash
   artifact bash -c 'for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
       python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
         --output-dir results/paper_results/recomputed/selected_centers/$ds || exit 1
     done'
   ```
4. **Choose the radius, compute the bounds, draw the figure and compare it with the paper**: `paper_results/figure2_biloc_selected_centers.py`
   ```bash
   artifact python -m paper_results.figure2_biloc_selected_centers
   ```

- [ ] `[PASS] Figure 2 (selected-center local-family certificates): 12 values match the paper, 0 differ`

## Four-byte key recovery (Section 5.3, full version)

Checks a median four-byte rank of 0 and a success rate at least the printed one (191/250, 206/250 and 221/250) at 500, 1,000 and 2,000 traces (6 values in Level D). Level A checks the archived rate at 500 traces (1 value).

1. **Download the raw ASCAD traces** (4.4 GB, see [Before you start](#before-you-start)): `multibyte_composition/fetch_raw_ascad.py`
2. **Find the trace points that leak each key byte**: `multibyte_composition/scan_raw_ascad_pois.py`
   ```bash
   artifact python -m multibyte_composition.scan_raw_ascad_pois \
     --output-dir results/paper_results/recomputed/multibyte_composition/poi_scan
   ```
3. **Train one attacker for each of eight key bytes** (0, 1, 2, 3, 4, 5, 8 and 15): `multibyte_composition/train_raw_ascad_byte_attacker.py`
   ```bash
   artifact bash -c 'R=results/paper_results/recomputed/multibyte_composition
     for b in 0 1 2 3 4 5 8 15; do
       python -m multibyte_composition.train_raw_ascad_byte_attacker --target-byte $b \
         --scan-topk $R/poi_scan/raw_ascad_poi_scan_topk.csv --output-dir $R/byte_attackers || exit 1
     done'
   ```
4. **Combine the byte attackers to recover four key bytes**, over 250 attack campaigns: `multibyte_composition/compose_four_byte_recovery.py`
   ```bash
   artifact python -m multibyte_composition.compose_four_byte_recovery \
     --input-dir results/paper_results/recomputed/multibyte_composition/byte_attackers \
     --campaigns 250 --trace-counts 1,2,5,10,20,50,100,200,500,1000,2000
   ```
5. **Build the table and compare it with the paper** (also builds the temperature table): `paper_results/fullversion_tables.py`
   ```bash
   artifact python -m paper_results.fullversion_tables
   ```

- [ ] `[PASS] Four-byte composition (full version): 6 values match the paper, 0 differ`

## PI and EPI under temperature rescaling (Section 5.4, full version)

Checks the raw PI variation, an EPI variation below `1e-8` bits, and unchanged predicted labels at every temperature for the six TCHES20 benchmarks (18 values).

1. **For one model per benchmark, compute PI, EPI and the predicted key-byte values at temperatures from 0.05 to 20** (Level D): `epi_comparison/epi_head_to_head.py`
   ```bash
   artifact bash -c 'T=datasets/raw/tches20/TCHES20V3_CNN_SCA
     for ds in ascad_desync_0 ascad_desync_50 ascad_desync_100 aes_hd aes_rd dpav4; do
       python -m epi_comparison.epi_head_to_head --dataset $ds --dataset-dir $T/datasets \
         --tches20-src-dir $T/src --models-dir $T/models \
         --output-dir results/paper_results/recomputed/epi_comparison/main || exit 1
     done'
   ```
2. **Build the table and compare it with the paper** (also builds the four-byte table): `paper_results/fullversion_tables.py`
   ```bash
   artifact python -m paper_results.fullversion_tables
   ```

- [ ] `[PASS] PI/EPI temperature sweep (full version): 18 values match the paper, 0 differ`

## Gaussian oracle (Section 5.1, full version)

Checks the success rate of the Bayes rule and of the learned Gaussian template on synthetic traces with a known posterior (2 values).

1. **Run the synthetic test where the best possible attacker is known** (three seeds): `bayes_oracle/gaussian_oracle_suite.py`
   ```bash
   artifact python -m bayes_oracle.gaussian_oracle_suite --output-dir results/paper_results/gaussian_oracle
   ```
2. **Compare with the paper**: Level A runs step 1 again, then checks every result: `paper_results/reproduce.py`
   ```bash
   artifact python -m paper_results.reproduce archived
   ```

- [ ] `[PASS] Synthetic Gaussian oracle (full version, Bayes-reference table): 2 values match the paper, 0 differ`

## By level

Each level runs the steps above in order and ends by running every builder, so later levels add to earlier ones.

| Level | Command | Steps it runs | Time |
|---|---|---|---|
| A | `python -m paper_results.reproduce archived` | Gaussian oracle, then every builder on the archived counts | 1 min |
| B | `python -m paper_results.reproduce tches20` | Table 7 steps 1–2 (with Table 8 step 1), Table 9 steps 1–2, Figure 2 step 1 | 8–15 min |
| C | `python -m paper_results.reproduce estranet` | Table 7 step 3 (with Table 8 step 2) | 20–70 min |
| C | `python -m paper_results.reproduce estranet --suites ches_windows` | Figure 1 step 1 | 34 min |
| C | Figure 2 steps 2–3, then `python -m paper_results.reproduce report` | Figure 2 steps 2–3 | 18 min |
| D | `python -m paper_results.reproduce fullversion` | four-byte steps 1–4, temperature sweep step 1 | about 20 min plus download |

Figure 1 steps 2–3 have no level of its own; run them, then `python -m paper_results.reproduce report`.

The Transformer checkpoints can also be retrained on a GPU ([guide 12](guides/pdf/12_gpu_docker.pdf)); the checks above use the released checkpoints.
