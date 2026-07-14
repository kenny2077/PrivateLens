# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS builder

ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ARG PRIVATELENS_EXTRAS=full
ARG PIP_INDEX_URL=https://pypi.org/simple

RUN if [ "$PRIVATELENS_EXTRAS" != "core" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN python -m pip install --no-cache-dir "uv==0.11.28"
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY privatelens/ ./privatelens/

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$PRIVATELENS_EXTRAS" = "core" ]; then \
        UV_INDEX_URL="$PIP_INDEX_URL" uv sync --locked --no-dev --no-editable; \
    else \
        UV_INDEX_URL="$PIP_INDEX_URL" uv sync --locked --no-dev --no-editable \
            --extra "$PRIVATELENS_EXTRAS"; \
    fi


FROM python:3.11-slim@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3 AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system privatelens \
    && useradd --system --gid privatelens --home-dir /app privatelens

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY scripts/ ./scripts/

RUN mkdir -p /data /photos /app/.privatelens/models /app/.privatelens/thumbnails \
    && chown -R privatelens:privatelens /data /photos /app

ENV PRIVATELENS_DATA_DIR=/data
ENV PRIVATELENS_MODEL_CACHE_DIR=/app/.privatelens/models
ENV PRIVATELENS_THUMBNAIL_DIR=/app/.privatelens/thumbnails
ENV PRIVATELENS_LOCAL_ONLY=true
ENV PRIVATELENS_OLLAMA_URL=http://ollama:11434

VOLUME ["/data", "/photos"]
EXPOSE 8000

USER privatelens

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/api/health >/dev/null || exit 1

CMD ["python", "-m", "privatelens.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
