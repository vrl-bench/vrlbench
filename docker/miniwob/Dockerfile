# The resolved base digest and installed dependencies can be recorded with docker inspect.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git fonts-liberation libcairo2 libpango-1.0-0 libasound2 libatk-bridge2.0-0 \
    libatk1.0-0 libatspi2.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 \
    libglib2.0-0 libnspr4 libnss3 libx11-6 libxcb1 libxcomposite1 libxdamage1 \
    libxext6 libxfixes3 libxkbcommon0 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/vrlbench
COPY src/vrlbench/requirements/runner-py312.lock.txt /tmp/runner-py312.lock.txt
RUN python -m pip install -r /tmp/runner-py312.lock.txt
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES ./
COPY src/ ./src/
RUN umask 022 && python -m pip install --no-deps . \
    && vrlbench setup miniwob --root /opt/miniwob
COPY examples/configs/ ./examples/configs/
RUN git config --system --add safe.directory /opt/miniwob/Synapse \
    && useradd --create-home --uid 1000 evaluator && mkdir /runs && chown evaluator /runs
USER evaluator
WORKDIR /runs
ENTRYPOINT ["vrlbench"]
CMD ["--help"]
