# Container for the return-risk scoring API.
#
# Two stages so the runtime image carries no build toolchain: wheels are built
# once and installed into a clean base. The image ships the package but not the
# trained model - mount `models/` at run time, which keeps the image immutable
# across retrains and avoids baking data into a layer.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# README.md is copied because pyproject.toml declares it as the package readme;
# without it the wheel build fails.
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels -r requirements.txt \
    && pip wheel --wheel-dir /wheels --no-deps .

# ---------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RETURNS_CLV_MODEL_PATH=/app/models

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

COPY api ./api
COPY configs ./configs

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
