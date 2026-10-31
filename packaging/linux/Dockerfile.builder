# Build in a rootless container; never install dependencies on the operator host.
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.96.0
# Build context is packaging/linux only, never the full dirty checkout.
COPY build-deps.txt /tmp/build-deps.txt
RUN apt-get update \
    && xargs -r apt-get install -y --no-install-recommends < /tmp/build-deps.txt \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain ${RUST_VERSION} \
    && rm /tmp/rustup.sh
ENV PATH=/opt/cargo/bin:$PATH
ENV CARGO_BUILD_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 RAYON_NUM_THREADS=2
WORKDIR /work
