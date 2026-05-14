FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /artifact

# libgomp1 supports common scientific Python wheels. fonts-dejavu-core keeps
# matplotlib figure generation predictable in headless environments.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt environment.yml setup.py README.md LICENSE CITATION.cff sitecustomize.py ./
COPY src ./src
COPY scripts ./scripts
COPY docs ./docs
COPY configs ./configs
COPY legacy ./legacy
COPY data ./data
COPY checkpoints ./checkpoints
COPY outputs/README.md ./outputs/README.md
COPY outputs/precomputed ./outputs/precomputed

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -e .

RUN chmod +x scripts/*.sh

CMD ["bash", "scripts/quick_test.sh"]
