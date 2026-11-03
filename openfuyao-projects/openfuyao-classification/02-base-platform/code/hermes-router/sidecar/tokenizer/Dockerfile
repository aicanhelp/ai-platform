# syntax=docker/dockerfile:latest

#######################################################################
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#######################################################################

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV_SYSTEM_PYTHON installs into the image interpreter so the runtime stage can
# still copy /usr/local/lib/python3.12/site-packages unchanged.
# UV_LINK_MODE=copy avoids bind-mounting the venv into the cache mount.
# UV_TORCH_BACKEND=cpu resolves torch + torchvision/torchaudio to CPU wheels so
# no nvidia-* CUDA packages are pulled in (CPU-only sidecar image).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cpu

WORKDIR /build

# build-essential is only needed when a dependency lacks a prebuilt wheel.
# All current dependencies (vllm, torch, grpcio, transformers, ...) ship
# manylinux wheels, so this layer is effectively dead weight but kept as a
# fallback for future deps.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer: depends ONLY on pyproject.toml, never on src.
# This is the layer that makes iterative builds fast -- editing source no
# longer re-resolves and reinstalls the full dependency tree (vllm + torch).
# uv.lock is intentionally gitignored (see .gitignore), so resolution happens
# at build time from pyproject.toml -- seconds with uv, and the layer only
# rebuilds when pyproject.toml changes. The cache mount persists uv's wheel
# cache so dependency changes reuse cached wheels instead of re-downloading.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip compile pyproject.toml -o requirements.txt \
    && uv pip install --system -r requirements.txt

# Source layer: dependencies are already installed, so this only installs
# the project package itself (no-deps) -- seconds, not minutes.
COPY src ./src
RUN uv pip install --system --no-deps .


FROM python:3.12-slim AS runtime

# HOME=/ puts the HF default cache ($HOME/.cache/huggingface) under /.cache,
# which is exactly where the kserve scheduler preset mounts its writable
# tokenizer-cache volume. TOKENIZER_CACHE_DIR pins it explicitly.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/ \
    TOKENIZER_CACHE_DIR=/.cache/huggingface/hub

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 nonroot \
    && useradd --system --uid 65532 --gid 65532 --home-dir / --no-create-home nonroot \
    && mkdir -p /mnt/models /tmp/tokenizer "$TOKENIZER_CACHE_DIR" \
    && chown -R 65532:65532 /mnt/models /tmp/tokenizer /.cache
# /mnt/models is pre-created (empty) so runtimes that reject a missing cwd
# tolerate the kserve preset's `workingDir: /mnt/models`; hermes ignores it.
# /tmp/tokenizer is the default UDS directory (covered by the preset's
# tokenizer-uds emptyDir mount when deployed via kserve).

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

USER 65532:65532

ENTRYPOINT ["tokenizer-sidecar"]