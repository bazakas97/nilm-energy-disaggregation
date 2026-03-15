FROM python:3.11-slim

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" && \
    pip install -r /app/requirements.txt

COPY . /app

# The compose setup mounts the project at /workspace, so run from there by default.
WORKDIR /workspace

ENTRYPOINT ["python", "run.py"]
CMD ["--config", "configs/active/release_eval.yaml"]
