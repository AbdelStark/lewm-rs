# syntax=docker/dockerfile:1.7

FROM rust:1.85.0-bookworm AS builder

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

LABEL org.opencontainers.image.source="https://github.com/AbdelStark/lewm-rs" \
      org.opencontainers.image.description="LEWM training runtime for Hugging Face Jobs" \
      org.opencontainers.image.licenses="MIT"

ENV HF_HOME=/tmp/hf \
    HDF5_PLUGIN_PATH=/usr/local/lib/python3.11/dist-packages/hdf5plugin/plugins \
    RUST_LOG=lewm=info,burn=info \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
        tar \
        zstd \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "huggingface_hub==${HF_CLI_VERSION}" \
        "hdf5plugin==6.0.0" \
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

CMD ["lewm-train", "--help"]
