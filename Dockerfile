# API image: FastAPI + the agent runtime. CPU-only torch comes from the lockfile (ADR-009).
# Data (panel, book, runs) is mounted at /app/data; secrets arrive as environment variables.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev && rm -rf /root/.cache

# Bake the retrieval embedding model so the container needs no Hugging Face access at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5', device='cpu')" \
    && chmod -R a+rX /opt/hf

COPY configs ./configs
COPY corpus ./corpus
COPY reports/metrics ./reports/metrics
COPY src ./src

RUN useradd --uid 1000 --create-home app && mkdir -p data && chown app data corpus
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# nginx serves the API under /api and strips the prefix; root-path keeps /docs working there.
CMD ["uvicorn", "riskgraph.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api"]
