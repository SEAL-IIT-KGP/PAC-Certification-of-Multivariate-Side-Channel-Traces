# Guides

One short PDF per folder: what it shows, what you need, the commands, and the terminal output you should see. The outputs were recorded in the Docker image on a Linux workstation (x86-64, CPU only). The same commands also passed every check in Docker on a Mac (Apple M3 Pro); guide 00 lists the times on both.

| Guide | Folder |
|---|---|
| [00_start_here.pdf](pdf/00_start_here.pdf) | overview, hardware, setup, downloads |
| [01_paper_results.pdf](pdf/01_paper_results.pdf) | `paper_results/`: Levels A to D |
| [02_bayes_oracle.pdf](pdf/02_bayes_oracle.pdf) | `bayes_oracle/` |
| [03_bi_suite_8datasets.pdf](pdf/03_bi_suite_8datasets.pdf) | `bi_suite_8datasets/` |
| [04_bi_scope_certificates.pdf](pdf/04_bi_scope_certificates.pdf) | `bi_scope_certificates/` |
| [05_epi_comparison.pdf](pdf/05_epi_comparison.pdf) | `epi_comparison/` |
| [06_multibyte_composition.pdf](pdf/06_multibyte_composition.pdf) | `multibyte_composition/` |
| [07_dimension_stability.pdf](pdf/07_dimension_stability.pdf) | `dimension_stability/` |
| [08_locality_diagnostics.pdf](pdf/08_locality_diagnostics.pdf) | `locality_diagnostics/` |
| [09_attacker_scope_diagnostics.pdf](pdf/09_attacker_scope_diagnostics.pdf) | `attacker_scope_diagnostics/` |
| [10_model_training.pdf](pdf/10_model_training.pdf) | `model_training/` |
| [11_core_datasets_scripts.pdf](pdf/11_core_datasets_scripts.pdf) | `core/`, `datasets/`, `scripts/`, `docs/` |

## Rebuilding the guides

| Path | Role |
|---|---|
| `chapters/*.md` | sources; `tools/build_pdf.sh [NN]` builds the PDFs (pandoc and XeLaTeX) |
| `transcripts/<guide>/<step>.txt` | recorded terminal output embedded in the chapters |
| `tools/plans.py`, `tools/record.py <guide> --mode docker` | the commands of each step, and the recorder that reruns them in the Docker image (`--only STEP`, `--force`) |
| `tools/compact_transcripts.py` | shortens transcripts for the page |
| `images/`, `tools/collect_figures.sh` | figures shown in the guides, copied from `results/` |
| `tools/check_margins.py` | checks that no line runs off a page |
