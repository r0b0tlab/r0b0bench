FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    R0B0BENCH_OUT=/out

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE AGENTS.md ./
COPY src ./src
COPY docs ./docs

RUN pip install --no-cache-dir -e .

RUN useradd -m -u 10001 bench \
 && mkdir -p /out /data \
 && chown -R bench:bench /app /out /data
USER bench

ENTRYPOINT ["r0b0bench"]
CMD ["profiles"]
