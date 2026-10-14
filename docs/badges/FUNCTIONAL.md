# Artifacts Functional

1. Install Docker.
2. Get the code:
   ```bash
   git clone https://github.com/K1015/PAC-Certification-of-Multivariate-Side-Channel-Traces.git pac-artifact
   cd pac-artifact
   ```
3. Build the image (1 to 3 minutes):
   ```bash
   docker build -t bi-certification .
   ```
4. Run the quick check (about 1 minute, no download):
   ```bash
   mkdir -p results
   docker run --rm -v "$PWD/results:/artifact/results" bi-certification
   ```
5. Read the end of the output: every line of the summary says `matches the paper`.
6. Open `results/paper_results/`: it holds the paper's tables (`.csv`, `.md`) and figures (`.pdf`, `.png`).
7. Run the code on real traces and models (8 to 15 minutes, 0.5 GB download):
   ```bash
   mkdir -p datasets/raw checkpoints external
   artifact() {
     docker run --rm -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
       -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
       bi-certification "$@"
   }
   artifact paper_results/fetch_tches20.sh
   artifact python -m paper_results.reproduce tches20
   ```
8. Read the end of the output again: every line of the summary says `matches the paper`.
9. Open [`REPRODUCTION_CHECKLIST.md`](../REPRODUCTION_CHECKLIST.md): each table and figure is listed with the code that makes it.
