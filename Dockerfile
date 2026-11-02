# syntax=docker/dockerfile:1
# Dev/parity image: Python 3.11 + R 4.x + R fda 6.3.0 + rpy2.
FROM rocker/r-ver:4.6.2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    R_HOME=/usr/local/lib/R

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        build-essential libffi-dev libpcre2-dev liblzma-dev libbz2-dev \
        libicu-dev zlib1g-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin R fda 6.3.0 from the CRAN archive (clean-room: used only to produce golden files).
RUN Rscript -e 'install.packages(c("jsonlite", "remotes"), repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("fda", version = "6.3.0", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'stopifnot(packageVersion("fda") == "6.3.0")'

RUN useradd -m -u 1000 dev
WORKDIR /work
COPY --chown=dev:dev pyproject.toml README.md LICENSE ./
COPY --chown=dev:dev src ./src
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install -e ".[dev,plot,pandas,io]"
USER dev
CMD ["bash"]
