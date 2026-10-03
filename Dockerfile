# Container for reproducing the tables and figures of
# "Bounded Information: PAC Certification of Multivariate Side-Channel Traces".
#
#   docker build -t bi-certification .
#   docker run --rm -v "$PWD/results:/artifact/results" bi-certification
#
# The default command rebuilds Tables 7-9 and Figures 1-2 from the archived
# counts. See paper_results/README.md for the runs that start from the traces.
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2

# git, zip, and unzip are used by the dataset fetch scripts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git zip unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt

COPY . .

CMD ["python", "-m", "paper_results.reproduce", "archived"]
