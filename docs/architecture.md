# Arabic Legal RAG — Final MLOps Architecture

## 1. Overview

The Arabic Legal RAG system is designed as an end-to-end MLOps pipeline for question answering over the Egyptian Civil Code.

The architecture separates the system into the following layers:

1. **Data and ingestion**
2. **Retrieval**
3. **Generation**
4. **Serving**
5. **Evaluation and experiment tracking**
6. **Observability and monitoring**
7. **CI/CD and containerization**
8. **Optimization and future deployment**

The design preserves article-level legal traceability throughout the pipeline so that generated answers can be linked back to specific Egyptian Civil Code articles and source pages.

---

## 2. Final Architecture

```text
                         ┌──────────────────────────┐
                         │   Egyptian Civil Code    │
                         │   Bilingual PDF (1948)   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      DVC / Versioning     │
                         │   Source + Derived Data   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Extraction Pipeline     │
                         │       PyMuPDF             │
                         │  Column-aware extraction  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Validation & Normalization│
                         │ Pydantic + Arabic cleanup │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Structured Article Corpus │
                         │      1,149 records        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Article-Level Chunking  │
                         │  Arabic + English text   │
                         │  + legal metadata        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Multilingual E5 Model   │
                         │ intfloat/multilingual-    │
                         │        e5-base            │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       FAISS Index         │
                         │      IndexFlatIP          │
                         │       768 dimensions      │
                         └────────────┬─────────────┘
                                      │
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
       ┌─────────────────────┐                  ┌─────────────────────┐
       │      FastAPI        │                  │      BentoML        │
       │       :8000         │                  │       :3000         │
       │      POST /ask      │                  │    LegalRAGService  │
       └──────────┬──────────┘                  └──────────┬──────────┘
                  │                                        │
                  └────────────────┬───────────────────────┘
                                   ▼
                         ┌──────────────────────────┐
                         │        Retriever         │
                         │   E5 query embedding     │
                         │     + FAISS search       │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     Context Builder      │
                         │ Article citation         │
                         │ Source page              │
                         │ Arabic + English text    │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       LLM Generator      │
                         │  OpenAI-compatible API   │
                         │       Groq / Qwen        │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Grounded RAG Answer     │
                         │ Arabic / English response │
                         │ + Article-level sources  │
                         └──────────────────────────┘


       ┌──────────────────────────────────────────────────────────────┐
       │                 EVALUATION & TRACKING                        │
       │                                                              │
       │  RAGAS ───────────────► MLflow                              │
       │  Retrieval metrics      Experiment tracking                │
       │  Faithfulness           Parameters                          │
       │  Context Precision      Metrics                             │
       │  Context Recall         Artifacts                           │
       └──────────────────────────────────────────────────────────────┘


       ┌──────────────────────────────────────────────────────────────┐
       │                    OBSERVABILITY                             │
       │                                                              │
       │  FastAPI /ask ─────────► Langfuse                           │
       │                           │                                  │
       │                           ├── Retrieval trace               │
       │                           ├── Retrieved articles             │
       │                           ├── Similarity scores              │
       │                           ├── LLM generation                 │
       │                           └── Prompt / response               │
       │                                                              │
       │  FastAPI /metrics ──────► Prometheus ──────► Grafana        │
       │                           │                   │              │
       │                           │                   ├── Request rate│
       │                           │                   ├── Errors     │
       │                           │                   ├── Retrieval  │
       │                           │                   │   latency    │
       │                           │                   ├── LLM latency│
       │                           │                   └── Availability│
       └──────────────────────────────────────────────────────────────┘


       ┌──────────────────────────────────────────────────────────────┐
       │                       CI / CD                                │
       │                                                              │
       │  GitHub Repository                                          │
       │          │                                                   │
       │          ▼                                                   │
       │  GitHub Actions                                             │
       │          │                                                   │
       │          ├── Ruff lint                                      │
       │          ├── Pytest                                         │
       │          ├── Docker build                                   │
       │          └── Docker image push                              │
       │                    │                                         │
       │                    ▼                                         │
       │                  GHCR                                        │
       │                    │                                         │
       │                    ▼                                         │
       │              Deployable Image                                │
       └──────────────────────────────────────────────────────────────┘
```

---

## 3. Data and Ingestion Layer

The source dataset is the bilingual Egyptian Civil Code PDF.

The ingestion pipeline converts the raw PDF into structured article-level records.

### Components

- PyMuPDF
- Article header detection
- Arabic numeral normalization
- Arabic text normalization
- Pydantic validation
- Repealed-article detection
- DVC versioning

### Data flow

```text
PDF
 ↓
Page extraction
 ↓
Column separation
 ↓
Article detection
 ↓
Arabic/English text extraction
 ↓
Normalization
 ↓
Validation
 ↓
Structured JSON
```

The resulting corpus contains 1,149 article-level records.

Each record preserves:

```text
article_number
book
chapter
section
topic_ar
topic_en
text_ar
text_en
source_page
citation
is_repealed
```

The original PDF and derived corpus are version-controlled as part of the data pipeline.

---

## 4. Retrieval Layer

The retrieval system uses multilingual semantic search.

### Embedding model

```text
intfloat/multilingual-e5-base
```

Embedding dimension:

```text
768
```

E5 uses different prefixes for documents and queries:

```text
passage: <legal article>
query: <user question>
```

Embeddings are normalized before indexing.

### Vector store

```text
FAISS
IndexFlatIP
```

Because vectors are normalized, inner-product search corresponds to cosine similarity.

### Retrieval flow

```text
Question
   ↓
query: <question>
   ↓
E5 embedding
   ↓
FAISS similarity search
   ↓
Top-K articles
   ↓
Article metadata + legal text
```

The current baseline uses `k=5`.

The retrieval evaluation established:

```text
Recall@1 = 0.75
Recall@3 = 0.90
Recall@5 = 0.90
MRR      = 0.8167
```

---

## 5. Context and Generation Layer

Retrieved articles are transformed into an LLM-ready context.

The context contains:

- Article citation
- Source PDF page
- Arabic legal text
- English legal text
- Relevant metadata

The prompt layer instructs the generator to:

- Use only retrieved legal context.
- Avoid unsupported legal conclusions.
- Cite the relevant article.
- Respond in Arabic when the question is Arabic.
- Indicate when the retrieved context is insufficient.

This creates the following flow:

```text
Question
   ↓
Retriever
   ↓
Top-K Articles
   ↓
Context Builder
   ↓
Grounded Prompt
   ↓
LLM
   ↓
Answer + Article Sources
```

The generator is abstracted behind a common `Generator` interface, allowing the application to use either a mock generator or an API-based LLM.

---

## 6. Serving Layer

The RAG pipeline is exposed through two serving interfaces.

### FastAPI

```text
Port: 8000
```

Main endpoints:

```text
GET  /health
POST /ask
GET  /metrics
```

FastAPI is used as the primary application API.

### BentoML

```text
Port: 3000
```

BentoML packages the same RAG components into a separately deployable service.

The BentoML service reuses:

```text
E5 embedding model
FAISS vector store
Retriever
Context builder
Generator
```

This avoids maintaining two separate RAG implementations.

---

## 7. Evaluation Layer

### Batched RAGAS Evaluation

To reduce API rate-limit errors when evaluating the 50-question Arabic legal QA dataset, the project provides a batched evaluation script.

**Configuration**

- **LLM provider:** Groq API (OpenAI-compatible endpoint)
- **Evaluation model:** `qwen/qwen3.8-27b`
- **Evaluation dataset:** 50 legal questions with generated answers and retrieved contexts
- **Batch size:** 5 questions per batch
- **Delay between batches:** 60 seconds
- **Rate-limit retries:** Exponential backoff, starting at 60 seconds
- **Metrics:** Faithfulness, Context Precision, Context Recall, and Answer Relevancy

**Run the evaluation**

From the repository root, run:

```bash
uv run python scripts/evaluate_ragas_batched.py
```

The evaluator processes the dataset in batches, saves a checkpoint after each successfully completed batch, and retries detected rate-limit errors with increasing delays.

**Output files**

| File | Description |
|---|---|
| `data/evaluation/ragas_checkpoint.json` | Progress and per-question results saved after each successful batch |
| `data/evaluation/ragas_results_batched.json` | Final evaluation results, aggregate metrics, and per-question scores |
| `data/evaluation/ragas_responses.json` | Input questions, generated answers, retrieved contexts, and reference answers |

The batched evaluation writes to separate output files and does not overwrite the original `data/evaluation/ragas_results.json`.

**Evaluation validation**

The script reports numeric score counts for each metric after every batch. Review these counts to identify missing scores before interpreting aggregate results. An evaluation that finishes processing all questions does not necessarily mean every metric score is valid.

**Important limitation:** The current script saves checkpoints but does not automatically resume from them after interruption. Rerunning the script starts evaluation from the first batch. The delay and retry settings reduce request bursts but cannot guarantee that provider rate limits will not be reached.

RAG quality is evaluated using RAGAS.

The evaluation dataset contains legal questions with reference information.

The main evaluation metrics are:

```text
Faithfulness
Context Precision
Context Recall
```

The current baseline achieved:

```text
Faithfulness      = 1.0000
Context Precision = 0.8750
Context Recall    = 1.0000
```

The evaluation results are stored as artifacts and tracked using MLflow.

## Limitation
### Groq API Rate Limit During RAGAS Evaluation

During the batched RAGAS evaluation, the evaluation process encountered a **Groq API rate-limit error (HTTP 429)** while using the `qwen/qwen3.8-27b` model as the evaluator LLM.

**Error details:**
- **Provider:** Groq API
- **Model:** `qwen/qwen3.8-27b`
- **Rate-limit type:** Tokens per day (TPD)
- **Daily token limit:** 200,000
- **Tokens consumed at the time of the error:** 199,339
- **Tokens requested by the failed operation:** 2,423
- **Suggested wait time:** Approximately 12 minutes and 41 seconds

The request exceeded the remaining daily token allowance, resulting in a rate-limit exception. This interrupted the current evaluation batch before it could complete successfully.

**Mitigation:** The evaluation script processes questions in batches of five, introduces a 60-second delay between successful batches, and retries detected rate-limit errors using exponential backoff. However, these measures cannot prevent daily quota exhaustion. The evaluation must resume after sufficient quota becomes available or continue with an evaluator model or API plan that has adequate token capacity.

**Impact on evaluation results:** Only successfully completed batches should be included in the reported evaluation results. The checkpoint file preserves completed batches, but the interrupted batch must be verified before treating the full 50-question evaluation as complete. Final aggregate metrics should be reported only after all 50 questions have valid results for the four metrics: faithfulness, context precision, context recall, and answer relevancy.
---

## 8. MLflow Experiment Tracking

MLflow provides experiment-level tracking.

It records:

```text
Embedding model
LLM model
Number of evaluation samples
Evaluation type
Faithfulness
Context precision
Context recall
Evaluation artifacts
```

The purpose of MLflow is to compare configurations over time.

For example:

```text
Baseline
   ↓
Retrieval improvement
   ↓
Prompt improvement
   ↓
Reranking
   ↓
Quantized model
```

Each experiment can be compared using the same evaluation dataset.

---

## 9. Langfuse Observability

Langfuse provides request-level observability.

Each `/ask` request can produce a trace containing:

```text
rag-ask
 ├── retrieval
 │    ├── question
 │    ├── top_k
 │    ├── article numbers
 │    └── similarity scores
 │
 └── llm-generation
      ├── model
      ├── prompts
      └── generated answer
```

This allows debugging of individual requests.

For example, if an answer is incorrect, the trace can help determine whether the problem originated from:

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

## 10. Production Monitoring

The FastAPI application exposes Prometheus metrics.

Custom metrics include:

```text
rag_requests_total
rag_errors_total
rag_retrieval_latency_seconds
rag_llm_latency_seconds
```

Prometheus periodically scrapes:

```text
/metrics
```

Grafana visualizes the collected metrics.

### Monitoring flow

```text
FastAPI
   │
   │ /metrics
   ▼
Prometheus
   │
   │ PromQL
   ▼
Grafana
```

The monitoring dashboard tracks:

- Request rate
- Error rate
- Retrieval latency
- LLM latency
- API latency
- API availability

---

## 11. Containerization

Docker packages the application and its runtime dependencies.

The image contains:

```text
Python 3.12
uv dependencies
FastAPI application
Structured corpus
FAISS vector store
E5 model
```

The E5 model is downloaded during image construction rather than at container startup.

The container exposes:

```text
8000
```

The Docker image can therefore run independently of the local Python environment.

---

## 12. CI/CD Pipeline

GitHub Actions automates validation and container publishing.

The intended pipeline is:

```text
Git Push / Pull Request
        ↓
GitHub Actions
        ↓
Install dependencies
        ↓
Ruff
        ↓
Pytest
        ↓
Docker Build
        ↓
Docker Push
        ↓
GHCR
```

The container registry provides a deployable version of the application image.

Authentication uses GitHub Actions' automatically provided:

```text
GITHUB_TOKEN
```

with:

```yaml
permissions:
  contents: read
  packages: write
```

No personal access token is required for the workflow.

---

## 13. Security and Secrets

Sensitive credentials are never embedded into the Docker image.

Examples include:

```text
GROQ_API_KEY
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
```

Local development uses `.env`.

CI/CD uses GitHub Actions Secrets.

The `.env` file must not be committed to the repository.

---

## 14. Optimization Layer

The architecture is designed to support later optimization without changing the core RAG pipeline.

Planned optimization components include:

### vLLM

Replace API-based generation with locally served inference.

```text
RAG
 ↓
vLLM
 ↓
Local LLM
```

### AWQ Quantization

The generative model can later be quantized to AWQ 4-bit.

The comparison should include:

```text
Original model
      vs
AWQ-4bit model
```

with evaluation of:

- RAGAS scores
- Latency
- Resource usage

### Streaming

The generation layer can expose tokens progressively to the client.

### Load Testing

Locust can evaluate the service under concurrent load.

The target from the project checklist is:

```text
50 concurrent users
```

### Guardrails

Guardrails can be added to detect and block inappropriate or sensitive content before returning an answer.

---

## 15. End-to-End MLOps Flow

The complete lifecycle is:

```text
                DATA
                 │
                 ▼
        Egyptian Civil Code
                 │
                 ▼
        Extraction + DVC
                 │
                 ▼
        Structured Corpus
                 │
                 ▼
        Article-Level Chunks
                 │
                 ▼
          E5 Embeddings
                 │
                 ▼
           FAISS Index
                 │
                 ▼
             RETRIEVAL
                 │
                 ▼
          Context Builder
                 │
                 ▼
            GENERATION
                 │
                 ▼
             LLM API
                 │
                 ▼
        Grounded Answer
                 │
       ┌─────────┼──────────┐
       │         │          │
       ▼         ▼          ▼
    RAGAS     Langfuse   Prometheus
       │         │          │
       ▼         │          ▼
    MLflow       │       Grafana
                 │
                 ▼
          Observability


        CODE / DEPLOYMENT
                 │
                 ▼
          GitHub Repository
                 │
                 ▼
          GitHub Actions
                 │
          ┌──────┴──────┐
          ▼             ▼
       Lint/Test      Docker
                        │
                        ▼
                       GHCR
                        │
                        ▼
                    Deployment
```

---

## 16. Architecture Principles

The system follows several key design principles.

### Separation of concerns

Data processing, retrieval, generation, evaluation, observability, and deployment are separated into independent components.

### Reproducibility

The source corpus, extraction pipeline, experiments, and container environment are versioned or reproducibly generated.

### Article-level traceability

The article is the primary legal unit throughout retrieval and answer generation.

### Provider independence

The `Generator` abstraction allows the LLM provider to be replaced without rewriting the retrieval pipeline.

### Deployment independence

FastAPI and BentoML provide alternative serving mechanisms over the same RAG components.

### Observable production behavior

MLflow tracks experiments, Langfuse traces individual requests, and Prometheus/Grafana monitor service behavior.

### Incremental optimization

vLLM, AWQ, streaming, guardrails, and load testing can be introduced without redesigning the core retrieval pipeline.

---

## 17. Final Component Summary

| Layer | Technology | Responsibility |
|---|---|---|
| Source | Egyptian Civil Code PDF | Legal corpus |
| Versioning | DVC | Data and pipeline versioning |
| Extraction | PyMuPDF | PDF extraction |
| Validation | Pydantic | Schema and corpus validation |
| Chunking | Python | Article-level documents |
| Embeddings | multilingual-e5-base | Semantic representation |
| Retrieval | FAISS | Similarity search |
| API | FastAPI | Application serving |
| Packaging | BentoML | Deployable RAG service |
| Generation | Groq / OpenAI-compatible API | LLM generation |
| Evaluation | RAGAS | RAG quality |
| Experiment tracking | MLflow | Metrics and experiments |
| Tracing | Langfuse | RAG/LLM observability |
| Metrics | Prometheus | Service monitoring |
| Dashboards | Grafana | Monitoring visualization |
| Containerization | Docker | Reproducible runtime |
| CI/CD | GitHub Actions | Automated validation/build |
| Registry | GHCR | Container image storage |
| Load testing | Locust | Concurrent-load evaluation |
| Optimization | vLLM / AWQ | Efficient inference |

---



## 18. Architecture Status

### Implemented

- Bilingual legal PDF extraction
- Structured article-level corpus
- DVC pipeline
- Article-level chunking
- E5 embeddings
- FAISS retrieval
- FastAPI API
- Grounded RAG generation
- RAGAS evaluation
- MLflow tracking
- Langfuse tracing
- Prometheus metrics
- Grafana monitoring
- BentoML serving
- Docker containerization
- GitHub Actions CI
- GHCR container publishing

### Optimization / Remaining Work

- CI RAGAS quality gate
- Reproducible index rebuild inside CI
- 50-user Locust evaluation
- Batch re-indexing validation
- Canary deployment documentation
- Extended RAGAS evaluation
- RAGAS-to-MLflow trend tracking
- Alerting
- Embedding drift monitoring
- Token/cost monitoring
- Guardrails
- vLLM serving
- AWQ 4-bit quantization
- Original vs quantized evaluation
- Latency comparison

The architecture is intentionally modular so these remaining components can be added without changing the fundamental data, retrieval, or legal-grounding design.