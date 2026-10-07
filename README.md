# Arabic Legal Document Q&A — RAG

An end-to-end **Arabic Legal RAG system** for question answering over the Egyptian Civil Code.

The system retrieves relevant legal articles from a bilingual Arabic/English corpus and uses a grounded LLM to generate answers with **article-level legal citations**.

> **Important:** This project is an MLOps/engineering implementation for research and demonstration purposes. It is not a substitute for professional legal advice.

---

## Overview

Legal question answering requires high retrieval accuracy and strong protection against hallucination.

This project addresses that problem by combining:

- Bilingual Arabic/English legal document processing
- Article-level document chunking
- Multilingual semantic embeddings
- FAISS vector search
- Retrieval-augmented generation (RAG)
- Grounded LLM generation
- Article-level source citations
- RAGAS evaluation
- MLflow experiment tracking
- Langfuse RAG observability
- Prometheus/Grafana monitoring
- FastAPI serving
- BentoML serving
- Docker containerization
- GitHub Actions CI/CD

### High-level architecture

```text
Egyptian Civil Code PDF
          │
          ▼
   Extraction & Validation
          │
          ▼
  Article-Level Corpus
          │
          ▼
 Multilingual E5 Embeddings
          │
          ▼
      FAISS Index
          │
          ▼
       Retrieval
          │
          ▼
    Context Builder
          │
          ▼
      LLM Generator
          │
          ▼
 Grounded Legal Answer
 + Article-Level Sources
```

The complete implementation architecture is documented in:

[`docs/implementation.md`](docs/implementation.md)

---

# Features

### Legal document processing

- Bilingual Arabic/English Egyptian Civil Code corpus
- Article-level extraction
- Arabic text normalization
- Article hierarchy preservation
- Repealed article detection
- Source PDF page tracking
- Pydantic schema validation

### Retrieval

- `intfloat/multilingual-e5-base`
- 768-dimensional embeddings
- FAISS `IndexFlatIP`
- Normalized embeddings for cosine-similarity search
- Article-level retrieval
- Citation-preserving metadata

### Generation

- Grounded RAG prompting
- Arabic/English responses
- Article citations
- Configurable LLM provider
- OpenAI-compatible API support
- Generator abstraction for provider independence

### MLOps

- DVC data versioning
- MLflow experiment tracking
- RAGAS evaluation
- Langfuse tracing
- Prometheus metrics
- Grafana dashboards
- Docker
- BentoML
- GitHub Actions

---

# Project Structure

```text
arabic-legal-rag/
│
├── data/
│   ├── raw/
│   │   └── egyptian_civil_code.pdf
│   └── processed/
│       └── corpus_raw.json
│
├── src/
│   ├── extraction/
│   ├── ingestion/
│   ├── rag/
│   ├── api/
│   ├── evaluation/
│   ├── monitoring/
│   └── services/
│
├── scripts/
│   ├── prepare_corpus.py
│   ├── validate_corpus.py
│   ├── build_e5_index.py
│   └── ...
│
├── tests/
│
├── docs/
│   └── implementation.md
│
├── reports/
│
├── prometheus/
│
├── .github/
│   └── workflows/
│
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

---

# Requirements

Before starting, install:

- Python **3.10+**
- Git
- DVC
- Docker Desktop — required for containerized deployment
- An LLM API key if using API-based generation

The project uses `uv` for dependency management, but a standard Python virtual environment with `pip` is also supported.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/AyaMYousef/-LLM-RAG-Arabic-Legal-Document-Q-A.git
cd -LLM-RAG-Arabic-Legal-Document-Q-A
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install the project

Using `pip`:

```bash
pip install -e ".[dev]"
```

Or using `uv`:

```bash
uv sync
```

The project is installable as a Python package, so the source code can be imported directly after installation.

---

# Data Setup

The legal source document is managed using DVC.

Pull the tracked data:

```bash
dvc pull
```

The main source document is:

```text
data/raw/egyptian_civil_code.pdf
```

The processed article-level corpus is generated from the source document.

To reproduce the extraction pipeline:

```bash
dvc repro
```

To check the DVC state:

```bash
dvc status
```

---

# Build the Vector Index

If the FAISS index does not exist locally, rebuild it from the processed corpus:

```bash
uv run python scripts/build_e5_index.py
```

This creates:

```text
data/processed/vector_store_e5/
├── index.faiss
└── metadata.json
```

The index uses:

```text
Embedding model: intfloat/multilingual-e5-base
Dimension: 768
Vector store: FAISS IndexFlatIP
```

The vector store is a generated artifact and can be rebuilt from the tracked corpus.

---

# Configuration

Create a local environment file based on the example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

For API-based generation, configure the required LLM settings in `.env`.

Example:

```env
GENERATOR=api

LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_api_key
LLM_MODEL=qwen/qwen3.8-27b
```

For local development without an LLM API, the mock generator can be used:

```env
GENERATOR=mock
```

### Secrets

Never commit `.env` or API credentials.

Examples of secrets include:

```text
GROQ_API_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
```

---

# Run the API

Start FastAPI with:

```bash
uv run uvicorn src.api.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Health Check

Check the service:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "documents_indexed": 1149
}
```

`documents_indexed` represents the number of article-level records available to the retriever.

---

# Ask a Legal Question

Send a question to:

```text
POST /ask
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟\"}"
```

The request format is:

```json
{
  "question": "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟"
}
```

The response contains:

- Generated answer
- Article citations
- Source pages

Example:

```json
{
  "answer": "...",
  "sources": [
    {
      "article_number": 1,
      "citation": "Egyptian Civil Code, Article 1",
      "source_page": 1
    }
  ]
}
```

The system preserves the article as the primary legal citation unit rather than exposing internal chunk IDs.

---

# Run with Docker

Docker provides a reproducible runtime containing the application, dependencies, corpus, FAISS index, and embedding model.

## Build

```bash
docker build -t arabic-legal-rag:latest .
```

## Run

```bash
docker run --rm -p 8000:8000 arabic-legal-rag:latest
```

Then open:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

The Docker image uses the mock generator by default unless runtime LLM configuration is supplied.

---

# BentoML

The RAG pipeline can also be served through BentoML.

Start the service:

```bash
uv run bentoml serve src.services.bentoml_service:LegalRAGService
```

The BentoML service runs on:

```text
http://127.0.0.1:3000
```

Health check:

```text
http://127.0.0.1:3000/healthz
```

BentoML reuses the same core RAG components:

```text
E5
 ↓
FAISS
 ↓
Retriever
 ↓
Context Builder
 ↓
Generator
```

This avoids maintaining a separate RAG implementation for BentoML.

---

# Monitoring

The FastAPI application exposes Prometheus metrics through:

```text
GET /metrics
```

Important custom metrics include:

```text
rag_requests_total
rag_errors_total
rag_retrieval_latency_seconds
rag_llm_latency_seconds
```

Prometheus can scrape the application using the configuration in:

```text
prometheus/prometheus.yml
```

Grafana can then visualize the metrics.

### Monitoring architecture

```text
FastAPI
   │
   │ /metrics
   ▼
Prometheus
   │
   ▼
Grafana
```

The monitoring dashboard includes request rate, error rate, retrieval latency, LLM latency, API latency, and availability.

Detailed monitoring setup is documented in:

[`docs/implementation.md`](docs/implementation.md)

---

# Evaluation

The project evaluates retrieval and end-to-end RAG quality separately.

## Retrieval evaluation

The current embedding baseline uses:

```text
intfloat/multilingual-e5-base
```

on a fixed legal-question benchmark.

Current baseline:

| Metric | Score |
|---|---:|
| Recall@1 | 0.7500 |
| Recall@3 | 0.9000 |
| Recall@5 | 0.9000 |
| MRR | 0.8167 |

## RAG evaluation

RAGAS is used to evaluate generated answers and retrieved context.

The current baseline evaluation includes:

- Faithfulness
- Context Precision
- Context Recall

Current baseline:

| Metric | Score |
|---|---:|
| Faithfulness | 1.0000 |
| Context Precision | 0.8750 |
| Context Recall | 1.0000 |

Evaluation artifacts are stored under:

```text
data/evaluation/
```

MLflow is used to track evaluation runs and compare configurations.

For the complete evaluation methodology and implementation history, see:

[`docs/implementation.md`](docs/implementation.md)

---

# Observability

Langfuse provides request-level RAG tracing.

A typical trace contains:

```text
rag-ask
├── retrieval
│   ├── question
│   ├── top_k
│   ├── article numbers
│   └── similarity scores
│
└── llm-generation
    ├── model
    ├── prompts
    └── generated answer
```

This makes it possible to determine whether an issue originated from:

```text
Retrieval
     ↓
Context construction
     ↓
Prompt
     ↓
LLM generation
```

---

# MLOps

The project uses the following MLOps components:

| Component | Purpose |
|---|---|
| DVC | Data and pipeline versioning |
| MLflow | Experiment tracking |
| RAGAS | RAG evaluation |
| Langfuse | Request-level tracing |
| Prometheus | Metrics collection |
| Grafana | Monitoring dashboards |
| Docker | Reproducible runtime |
| BentoML | Model/RAG serving |
| GitHub Actions | CI/CD |
| GHCR | Container registry |
| Locust | Load testing |

---

# Testing

Run the automated test suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

The CI pipeline runs automated validation through GitHub Actions.

---

# CI/CD

The repository uses GitHub Actions for automated project validation.

The CI pipeline validates the codebase and builds the application container.

The deployment workflow is based on:

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ├── Lint
   ├── Tests
   └── Docker Build
          │
          ▼
        GHCR
```

The complete CI/CD implementation and current status are documented in:

[`docs/implementation.md`](docs/implementation.md)

---

# Repository Documentation

The main README is intentionally focused on **using and running the system**.

Detailed implementation documentation is available in:

### Implementation Documentation

[`docs/implementation.md`](docs/implementation.md)

Contains:

- Extraction implementation
- Corpus validation
- DVC pipeline
- Chunking design
- Embedding experiments
- FAISS retrieval
- RAG implementation
- RAGAS evaluation
- MLflow tracking
- Langfuse integration
- BentoML
- Prometheus
- Grafana
- Docker
- CI/CD
- Architecture
- Optimization work

### Architecture

[`docs/architecture.md`](docs/architecture.md)

Contains the system architecture and component relationships.

### Evaluation

[`docs/evaluation.md`](docs/evaluation.md)

Contains evaluation methodology and results.

---

# Quick Start

For the shortest path from clone to a running RAG API:

```bash
git clone https://github.com/AyaMYousef/-LLM-RAG-Arabic-Legal-Document-Q-A.git
cd -LLM-RAG-Arabic-Legal-Document-Q-A

python -m venv .venv
```

Activate the environment, then:

```bash
pip install -e ".[dev]"
dvc pull
uv run python scripts/build_e5_index.py
uv run uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Then send a question to:

```text
POST /ask
```

---

# Project Status

The project implements an end-to-end Arabic Legal RAG and MLOps architecture covering:

```text
Data
 ↓
Extraction
 ↓
Validation
 ↓
DVC
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieval
 ↓
Context
 ↓
LLM Generation
 ↓
FastAPI / BentoML
 ↓
RAGAS
 ↓
MLflow
 ↓
Langfuse
 ↓
Prometheus
 ↓
Grafana
 ↓
Docker
 ↓
CI/CD
```

For the detailed implementation record and completed project requirements, see:

[`docs/implementation.md`](docs/implementation.md)
