# Agenthon 2026 Track 2 submission image.
#
# Verb is an executable on PATH (no ENTRYPOINT), per SUBMISSION_CLI.md. CMD is the verb + --help
# so `docker run <img>` is self-describing; the harness overrides argv with the real invocation:
#
#   docker run --rm --network=none \
#     -v <unit-dir>:/input:ro -v <out>:/output \
#     track2-agent:latest \
#     forecast --panels /input/panels --text /input/text --asof 2024-06-28 \
#              --out /output/forecast.parquet
#
# In official `restricted`-network scoring, the same container reaches the house model endpoint
# through the organizer's audited proxy only ($MODEL_ENDPOINT / $MODEL_NAME / $HTTP_PROXY set by
# the harness); no vendor API key exists and none is baked into this image.

# Python 3.13 is required by qfbench2-common v2.6.1.  This digest is the
# linux/amd64 child manifest for python:3.13-slim-bookworm, verified against
# Docker Hub on 2026-10-17.  The base config declares no Docker VOLUME.
FROM python:3.13-slim-bookworm@sha256:f040863673aea2570c3ff6a5c3fb4c673a016cbc5375005ad145915922b6b78a

LABEL qfbench2.interface_version="2.0"
LABEL qfbench2.track="forecasting"
LABEL qfbench2.verb="forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# Git is required only while building because the shared toolkit is installed
# from its immutable public source commit.  All forecast and validation
# dependencies are installed at build time; the runtime never installs code.
RUN apt-get update \
 && apt-get install --no-install-recommends -y git \
 && pip install --no-cache-dir \
        "numpy==2.5.4" \
        "pandas==3.0.6" \
        "pyarrow==26.0.0" \
        "scikit-learn==1.9.1" \
        "jsonschema==4.26.0" \
        "scipy==1.18.1" \
        "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@03c50a0036aaf72675d353aef4c7c01ca86d1fcf#subdirectory=common" \
 && rm -rf /var/lib/apt/lists/*

COPY src/track2_agent /opt/track2_agent
ENV PYTHONPATH=/opt

RUN printf '#!/bin/sh\nexec python3 -m track2_agent.cli "$@"\n' > /usr/local/bin/forecast \
 && chmod +x /usr/local/bin/forecast

# Non-root: /input is read-only and /output is the only writable mount this image ever needs.
RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["forecast", "--help"]
