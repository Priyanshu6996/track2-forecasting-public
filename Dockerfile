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

FROM python:3.11-slim-bookworm

LABEL qfbench2.interface_version="2.0"
LABEL qfbench2.track="forecasting"
LABEL qfbench2.verb="forecast"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# Pinned, matching pyproject.toml's floors. Pinning here (not just in pyproject) is what makes a
# rebuild reproducible byte-for-byte in the dependency graph, independent of PyPI's current
# resolver state on the day of the build.
RUN pip install --no-cache-dir \
        "numpy==1.26.4" \
        "pandas==2.2.2" \
        "pyarrow==16.1.0" \
        "scikit-learn==1.4.2" \
        "jsonschema==4.22.0"

COPY src/track2_agent /opt/track2_agent
ENV PYTHONPATH=/opt

RUN printf '#!/bin/sh\nexec python3 -m track2_agent.cli "$@"\n' > /usr/local/bin/forecast \
 && chmod +x /usr/local/bin/forecast

# Non-root: /input is read-only and /output is the only writable mount this image ever needs.
RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["forecast", "--help"]
