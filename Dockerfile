# syntax=docker/dockerfile:1.7
#
# lewm-rs training runtime for Hugging Face Jobs.
#
# Build stage compiles `lewm-train` (release) against a pinned Rust toolchain.
# Runtime stage is a slim Debian image with the `huggingface_hub` CLI, hdf5plugin
# codecs (zstd / blosc / lz4 for PushT episodes), and the binary, executed as a
# non-root user under `tini` for clean signal handling.
#
# Image is published at `ghcr.io/abdelstark/lewm-rs`. HF Jobs default to
# `latest` for development; `scripts/launch_hf_job.py --image-tag` can pin a
# release tag at submission time without editing the YAML.

FROM rust:1.95.0-bookworm AS builder

WORKDIR /src

ENV CARGO_INCREMENTAL=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        git \
        pkg-config \
        python3 \
        zstd \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
COPY configs ./configs

RUN cargo build --locked --release -p lewm-train

FROM debian:bookworm-slim AS runtime

ARG HF_CLI_VERSION=1.8.0
ARG HDF5PLUGIN_VERSION=6.0.0
ARG NUMPY_VERSION=2.4.4
ARG BUILD_REVISION="unknown"
ARG BUILD_DATE="unknown"
ARG SOURCE_VERSION="unknown"

LABEL org.opencontainers.image.source="https://github.com/AbdelStark/lewm-rs" \
      org.opencontainers.image.description="lewm-rs training runtime for Hugging Face Jobs" \
      org.opencontainers.image.documentation="https://abdelstark.github.io/lewm-rs/" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="lewm-rs" \
      org.opencontainers.image.authors="Abdel <abdel@starkware.co>" \
      org.opencontainers.image.vendor="AbdelStark" \
      org.opencontainers.image.url="https://github.com/AbdelStark/lewm-rs" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${SOURCE_VERSION}" \
      org.opencontainers.image.base.name="docker.io/library/debian:bookworm-slim"

ENV HF_HOME=/tmp/hf \
    HDF5_PLUGIN_PATH=/usr/local/lib/python3.11/dist-packages/hdf5plugin/plugins \
    RUST_LOG=lewm=info,burn=info \
    PYTHONUNBUFFERED=1 \
    TINI_KILL_PROCESS_GROUP=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        tar \
        tini \
        zstd \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "huggingface_hub==${HF_CLI_VERSION}" \
        "hdf5plugin==${HDF5PLUGIN_VERSION}" \
        "numpy==${NUMPY_VERSION}" \
        "safetensors==0.5.3" \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /src/target/release/lewm-train /usr/local/bin/lewm-train
COPY configs ./configs
COPY jobs ./jobs
COPY python ./python

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin lewm \
    && mkdir -p /tmp/hf /tmp/data /tmp/out \
    && chown -R lewm:lewm /workspace /tmp/hf /tmp/data /tmp/out

USER lewm

# Liveness probe: validates the binary loads, the workspace is mounted, and the
# Python edge layer is importable. Cheap (<200 ms) and idempotent.
HEALTHCHECK --interval=60s --timeout=10s --start-period=15s --retries=2 \
    CMD lewm-train --version >/dev/null \
        && python -c "import huggingface_hub, hdf5plugin, numpy" >/dev/null \
        || exit 1

# `tini` reaps zombies and forwards SIGTERM during graceful shutdown. The image
# stays callable as a flexible wrapper: any HF Jobs `command:` override (e.g.
# `bash -lc "lewm-train train …"`) flows through tini unchanged.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["lewm-train", "--help"]
