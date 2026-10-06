FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GENERATOR=mock \
    LLM_MODEL=mock \
    HF_HOME=/opt/huggingface

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY src ./src
COPY data/processed/corpus_raw.json ./data/processed/corpus_raw.json
COPY data/processed/vector_store_e5 ./data/processed/vector_store_e5

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]