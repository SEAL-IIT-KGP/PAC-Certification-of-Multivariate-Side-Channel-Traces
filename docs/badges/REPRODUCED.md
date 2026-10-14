# Artifacts Reproduced

1. Do steps 1 to 3 and step 7 of [Artifacts Functional](FUNCTIONAL.md). Check that the output has `[PASS]` (the values match the paper) for Tables 7, 8 and 9 and Figure 2, and the line `80/80 recomputed per-model counts equal the archived counts` for each of the six TCHES20 benchmarks.
2. Download the Transformer models and two more trace sets (0.7 GB):
   ```bash
   artifact python -m paper_results.fetch_checkpoints
   artifact paper_results/fetch_estranet.sh
   artifact python -m paper_results.fetch_ascad_random_key
   ```
3. Download `CHES_Challenge.h5` (4 GB) from https://pace-tl.gitbook.io/ches-challenge-2025 and put it in `datasets/raw/ches_ctf_2025/`.
4. Recompute the Transformer rows of Tables 7 and 8 (20 to 70 minutes). Check `[PASS]` for Tables 7 and 8.
   ```bash
   artifact python -m paper_results.reproduce estranet
   ```
5. Recompute the BI line of Figure 1 (30 minutes to 2 hours, 12 GB of memory). Check `[PASS]` for Figure 1 and for its additional `delta = 1e-6` figure.
   ```bash
   artifact python -m paper_results.reproduce estranet --suites ches_windows
   ```
6. Recompute all twelve models of Figure 2 (15 to 30 minutes). Check `[PASS]` for Figure 2.
   ```bash
   artifact bash -c 'python -m bi_scope_certificates.prepare_aesrd_estranet_h5 &&
     for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
       python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
         --output-dir results/paper_results/recomputed/selected_centers/$ds; done &&
     python -m paper_results.reproduce report'
   ```
7. Recompute the two full-version results quoted in Section 5 (10 to 30 minutes, plus a 4.4 GB download). Check `[PASS]` for the four-byte composition and the PI/EPI temperature sweep.
   ```bash
   artifact python -m paper_results.reproduce fullversion
   ```
8. Compare the files in `results/paper_results/` with the paper. [`REPRODUCTION_CHECKLIST.md`](../REPRODUCTION_CHECKLIST.md) says which file belongs to which table or figure.
9. Optional, about 18 hours: recompute the MI and PI lines of Figure 1 with steps 2 and 3 of the Figure 1 section of the checklist.
