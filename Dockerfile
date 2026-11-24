# syntax=docker/dockerfile:1
# Glove detection API, CPU only. The model (22.5 MB, AGPL-3.0) is mounted read-only at
# run time (docker-compose.yml); fetch it first with tools/download_model.py.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1
# OpenCV (pulled in by Ultralytics) needs these shared libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY app/ app/
COPY services/ services/
COPY demo/ demo/
COPY detector.py utils.py detection_script.py run.py ./

RUN useradd --uid 1000 --create-home glove && mkdir -p models
USER glove

ENV PATH=/app/.venv/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    GLOVE_MODEL_PATH=/app/models/best_50epc.pt
EXPOSE 8004
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8004/health')" || exit 1
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8004"]
