FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/workspace/lunwen/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock

RUN python -m pip install --require-hashes -r /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock

RUN groupadd --gid 10001 researcher \
    && useradd --create-home --uid 10001 --gid 10001 --shell /bin/bash researcher \
    && install -d -o researcher -g researcher /workspace/lunwen /workspace/runtime

WORKDIR /workspace/lunwen
USER researcher

CMD ["bash"]
