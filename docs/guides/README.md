# Guides

One short PDF per folder of the repository. Each guide says what the folder shows, what you need
before you start, the commands to type, and the terminal output you should see. The outputs were
recorded inside the Docker image on a MacBook (Apple M3 Pro, 12 cores, 14 GB given to Docker),
CPU only.

| Guide | Folder |
|---|---|
| [00_start_here.pdf](pdf/00_start_here.pdf) | what the code does, hardware, setup with and without Docker, downloads |
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

## How the guides are made

- `chapters/*.md` are the sources; `tools/build_pdf.sh` builds the PDFs with pandoc and XeLaTeX
  (`tools/build_pdf.sh 03` builds one guide).
- `transcripts/<guide>/<step>.txt` are the recorded terminal outputs that the chapters embed.
  `tools/plans.py` lists the command of every step; `tools/record.py <guide> --mode docker`
  reruns the steps of a guide in the `bi-certification` image and rewrites its transcripts
  (`--only STEP`, `--force`); `tools/compact_transcripts.py` shortens them for the page.
- `images/` holds the regenerated figures shown in the guides (`tools/collect_figures.sh` copies
  them from `results/`), and `tools/check_margins.py` checks that no line runs off a page.
