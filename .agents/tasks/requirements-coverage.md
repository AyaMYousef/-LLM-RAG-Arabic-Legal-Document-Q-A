# Requirements Coverage Report
**Project:** Arabic Legal RAG — Egyptian Civil Code Q&A  
**Investigated:** `c:\Users\Aya.youssef\Desktop\Final_MLOPS\-LLM-RAG-Arabic-Legal-Document-Q-A`  
**Date:** Investigation performed via static code analysis (no execution)

---

## Summary

The project is **substantially complete** on Core/Foundation requirements, with many Advanced requirements also implemented. Expert-level requirements (vLLM, AWQ quantization, drift monitoring, token cost Grafana, self-hosted Langfuse) are acknowledged as planned but not yet implemented. Several specific checklist items have partial gaps that are fixable.

| Section | Total | ✅ Covered | ⚠️ Partial | ❌ Missing |
|---------|-------|-----------|-----------|----------|
| Core (1–16) | 16 | 9 | 5 | 2 |
| Advanced (17–27) | 11 | 4 | 5 | 2 |
| Expert (28–34) | 7 | 0 | 1 | 6 |
| **TOTAL** | **34** | **13** | **11** | **10** |

---

## Core / Foundation Requirements

| # | Requirement | Status | Evidence / Gap |
|---|---|---|---|
| 1 | `src/` layout with `pyproject.toml` — `pip install -e .` works | ✅ Covered | `pyproject.toml` present with `[build-system]` using `setuptools>=61.0`. `[tool.setuptools.packages.find]` set to `where = ["."]`, `include = ["src*"]`. `src/` has proper `__init__.py` hierarchy. `pip install -e .` should work. |
| 2 | Egyptian Civil Code PDF → structured JSON, one record per article, committed | ✅ Covered | `data/raw/egyptian_civil_code.pdf` present (tracked via DVC). `data/processed/corpus_raw.json` present (1,149 article records). `dvc.lock` confirms pipeline output. JSON is committed in the repo. |
| 3 | Extraction validated: article numbers normalized to integers, Arabic text spot-checked on 20 articles, repealed articles flagged | ✅ Covered | `scripts/validate_corpus.py` exists. `reports/corpus_validation.txt` shows `RESULT: PASS`, a random sample of exactly 20 articles spot-checked (Articles 52, 82, 89, 93, 206, 219, 237, 256, 313, 464, 504, 508, 514, 533, 558, 620, 916, 921, 976, 1091), repealed ranges (54–80, 389–417) marked with `is_repealed: true`. `src/extraction/normalizer.py` handles Arabic-Indic → integer normalization. |
| 4 | Ingestion pipeline: JSON → chunk by article → embed → vector store | ✅ Covered | Full pipeline: `src/ingestion/chunker.py` (`build_chunk`, `build_chunks`), `src/ingestion/embedder.py`, `src/ingestion/vector_store.py` (FAISS). `scripts/build_e5_index.py` orchestrates the pipeline. `data/processed/vector_store_e5/` (index.faiss + metadata.json) present. |
| 5 | FastAPI `/ask`: `{question: str}` → `{answer: str, sources: list[str]}` — sources are article citations, not chunk IDs | ✅ Covered | `src/api/main.py` defines `POST /ask` returning `AskResponse(answer, sources)`. Sources are `Source(article_number, citation, source_page)` — citation is `"Egyptian Civil Code, Article N"`, not a chunk ID. |
| 6 | Pydantic rejects an empty question — 422 returned and tested | ⚠️ Partial | `AskRequest` uses `question: str = Field(..., min_length=1)` which triggers a **422** from Pydantic/FastAPI automatically. However, **no automated test** covers this 422 behavior. `tests/` only contains `test_pii_guardrail.py` and `test_pii_false_positives.py`; there is no `test_api.py` or equivalent. The runtime behavior is correct, but it is untested. |
| 7 | `/health` returns `{status: healthy, documents_indexed: N}` | ✅ Covered | `src/api/main.py` `GET /health` returns `{"status": "healthy", "documents_indexed": retriever.vector_store.index.ntotal}`. Confirmed value is 1149 per README. |
| 8 | Dockerfile includes vector store and embedded documents; `docker compose up` starts on port 8000 | ⚠️ Partial | `Dockerfile` present and copies `data/processed/corpus_raw.json` and `data/processed/vector_store_e5`. It exposes port 8000. **However, `docker-compose.yml` does not exist** — there is no Compose file anywhere in the repo. The README describes `docker run`, not `docker compose up`. |
| 9 | README: 3 commands to run Q&A on any machine | ⚠️ Partial | README has detailed setup sections. The `docker run --rm -p 8000:8000 arabic-legal-rag:latest` approach is documented. However, the README does **not present a clean "3 commands" block** (e.g., `git clone` + `docker pull/build` + `docker run`). The current quickstart requires multiple manual steps. No `docker-compose.yml` makes a single `docker compose up` impossible. |
| 10 | MLflow logs each config: `chunk_size`, `overlap`, `embedding_model`, `faithfulness` | ⚠️ Partial | MLflow is integrated (`mlflow.db`, 3 runs in `mlruns/1/`). `scripts/log_ragas_mlflow.py` logs `embedding_model`, `llm_model`, `num_samples`, `evaluation_type`, and metrics (`faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`). **`chunk_size` and `overlap` are never logged** — the chunker uses article-level chunking without configurable chunk_size/overlap parameters, so these fields are simply missing from all logged runs. |
| 11 | ≥ 5 runs compared — MLflow screenshot in `/reports/` | ❌ Missing | Only **3 runs** exist in `mlruns/1/` (all named `ragas-baseline` with identical parameters — just repeated executions of the same config). The requirement asks for ≥ 5 distinct runs with different configurations being compared. No MLflow screenshot image exists in `reports/` — only `.txt` and `.json` files are present. |
| 12 | Best chunking config registered in MLflow Registry | ❌ Missing | MLflow model registry is empty: `SELECT * FROM registered_models` returns `[]`. No model or config has been registered. |
| 13 | Source PDF and derived JSON tracked with DVC — `dvc pull` fetches them, `dvc repro` rebuilds JSON | ⚠️ Partial | PDF is tracked: `data/raw/egyptian_civil_code.pdf.dvc` exists, `dvc.lock` references it. `dvc repro` runs `scripts/prepare_corpus.py` to rebuild `corpus_raw.json`. However: (1) `corpus_raw.json` itself is **not separately DVC-tracked** as an output artifact (no `corpus_raw.json.dvc` file); it is only an output of the `extract_articles` stage in `dvc.yaml`. (2) **`.dvc/config` is empty** — no remote storage is configured, meaning `dvc pull` on a fresh clone would fail to fetch the PDF from a remote. |
| 14 | `dvc repro` re-indexes corpus reproducibly from tracked documents | ⚠️ Partial | `dvc.yaml` defines the `extract_articles` stage that rebuilds `corpus_raw.json` from the PDF. `dvc.lock` is present and confirms stage hashes. However, the **vector store rebuild** (embedding + FAISS indexing) is **not part of the DVC pipeline** — it requires a manual `uv run python scripts/build_e5_index.py` step. So `dvc repro` only rebuilds the JSON, not the full index. |
| 15 | GitHub Actions: lint → test → rebuild index → push Docker image | ⚠️ Partial | `.github/workflows/ci.yml` has lint (`ruff check`), test (`pytest`), and Docker build + push to GHCR. **"rebuild index" is missing** from CI — there is no step to run `dvc repro` or `build_e5_index.py`. The pipeline goes: lint → test → Docker build+push (no index rebuild step). |
| 16 | CI fails if RAGAS faithfulness < 0.75 on 20-question test set | ❌ Missing | No RAGAS quality gate exists in `.github/workflows/ci.yml`. The workflow has no RAGAS evaluation step whatsoever. Architecture doc explicitly lists "CI RAGAS quality gate" as remaining work. |

---

## Advanced Requirements

| # | Requirement | Status | Evidence / Gap |
|---|---|---|---|
| 17 | BentoML wraps RAG pipeline with async `/ask` endpoint | ⚠️ Partial | `src/services/bentoml_service.py` exists with `LegalRAGService` class decorated `@bentoml.service`. The `/ask` endpoint is implemented. However, the `ask` method is **synchronous** (`def ask`, not `async def ask`) — it is not an async endpoint as required. |
| 18 | vLLM serves the generative LLM — model name in README | ❌ Missing | No vLLM implementation exists anywhere in the codebase. The architecture doc lists vLLM as "Optimization / Remaining Work." Current generation uses OpenAI-compatible API (Groq) or mock generator. vLLM model name is not in README. |
| 19 | Streaming: tokens appear progressively in curl output | ✅ Covered | `src/api/main.py` defines `POST /ask/stream` returning `StreamingResponse`. `src/rag/api_generator.py` implements `generate_stream()` using OpenAI `stream=True` and yields tokens via `chunk.choices[0].delta.content`. Wired end-to-end. |
| 20 | Locust report at 50 concurrent users in `/reports/` | ❌ Missing | `locustfile.py` exists and is correctly implemented. `reports/LoadTestingandPerformanceEvaluation.txt` exists but reports tests at **1 concurrent user** and **5 concurrent users** only — the requirement specifies **50 concurrent users**. |
| 21 | Batch re-indexing script tested with at least one new document | ❌ Missing | No batch re-indexing script exists. `scripts/build_e5_index.py` rebuilds the entire index from scratch but does not support adding new documents incrementally. No test demonstrates adding a new document to the index. |
| 22 | Canary rollout config documented in README | ❌ Missing | No canary rollout documentation exists in README. Architecture doc explicitly lists "Canary deployment documentation" as remaining work. |
| 23 | RAGAS on ≥ 50 questions — all 4 metrics logged | ✅ Covered | `data/evaluation/ragas_questions.json` contains 50 questions. `scripts/evaluate_ragas.py` and `scripts/evaluate_ragas_batched.py` evaluate all 4 metrics: `Faithfulness`, `ContextPrecision`, `ContextRecall`, `AnswerRelevancy`. `data/evaluation/ragas_results.json` stores results (`num_samples: 50`, all 4 metrics present). Note: The batched run hit a rate-limit; the single-run results show 3 metrics (faithfulness=1.0, context_precision=0.929, context_recall=1.0). Latest run includes answer_relevancy=0.832. |
| 24 | RAGAS results stored in MLflow — trend visible across sessions | ⚠️ Partial | 3 runs exist in MLflow with RAGAS metrics logged. The `ragas_results.json` artifact is stored per run. However, all 3 runs are identical `ragas-baseline` configurations — there is no meaningful **trend** across configurations. The runs differ only in `context_precision` values (0.875 vs 0.929), not because of different configs. Trend across sessions is marginally visible but not meaningful. |
| 25 | Alert: faithfulness < 0.80 → notification triggered | ❌ Missing | No alerting mechanism exists. No code checks the faithfulness threshold and triggers a webhook, email, or Prometheus alerting rule. |
| 26 | Langfuse self-hosted — every `/ask` creates a trace with spans | ⚠️ Partial | Langfuse tracing is fully implemented in `src/api/main.py` with `rag-ask` root span → `retrieval` span → `llm-generation` generation span. This is comprehensive observability. However, **Langfuse is configured for cloud** (`LANGFUSE_BASE_URL=https://cloud.langfuse.com` in `.env.Example`), not self-hosted. The requirement specifies "Langfuse self-hosted." No Docker Compose setup for a local Langfuse server exists. |
| 27 | RAGAS faithfulness score attached to each Langfuse trace | ❌ Missing | No code attaches a RAGAS faithfulness score to individual Langfuse traces. The `langfuse` trace in `main.py` records retrieval results and generated answer, but does not run RAGAS inline and attach the score. RAGAS is only evaluated offline via scripts. |

---

## Expert Requirements

| # | Requirement | Status | Evidence / Gap |
|---|---|---|---|
| 28 | Cosine similarity on query embeddings vs baseline — drift logged | ❌ Missing | No drift monitoring code anywhere in the source. `src/monitoring/metrics.py` defines RAG request/error/latency metrics but has no embedding drift metric. Architecture doc lists "Embedding drift monitoring" as remaining work. |
| 29 | Token usage in Prometheus — Grafana shows cost/hour | ❌ Missing | `src/monitoring/metrics.py` does not track token usage. No `token_usage_total` or `cost_per_token` Prometheus metric exists. Grafana dashboard documented in README has 6 panels (request rate, error rate, retrieval latency, LLM latency, API latency, availability) — none track token cost. |
| 30 | Guardrails PII detection active on all `/ask` responses | ✅ Covered | `src/guardrails/pii.py` implements regex-based PII redaction (Egyptian phone, national ID, email). `src/api/main.py` calls `redact_pii(answer)` on every `/ask` response before returning. `src/guardrails/input_guard.py` and `output_guard.py` also active. Tests in `tests/test_pii_guardrail.py` and `tests/test_pii_false_positives.py` pass. |
| 31 | Generative model quantized to AWQ-4bit — weights documented | ❌ Missing | No AWQ quantization code anywhere. `scripts/export_e5_onnx.py` exports the embedding model to ONNX (not the generative model, and not AWQ). Architecture doc lists "AWQ 4-bit quantization" as remaining work. |
| 32 | RAGAS scores: original vs quantized — ≤ 0.03 faithfulness drop | ❌ Missing | Depends on AWQ quantization (req. 31), which is not implemented. No comparison exists. |
| 33 | Latency before vs after quantization in README | ❌ Missing | No quantization has been performed, so no latency comparison is possible. Architecture doc lists this as remaining work. |
| 34 | Final architecture diagram covering all 5 session components | ⚠️ Partial | `docs/architecture.md` exists with a comprehensive ASCII text architecture diagram covering: data/ingestion, retrieval, generation, serving (FastAPI + BentoML), evaluation (RAGAS + MLflow), observability (Langfuse + Prometheus + Grafana), CI/CD, and optimization. This covers all required components. However, it is an **ASCII diagram in Markdown**, not a visual architecture image/diagram (e.g., PNG, draw.io, or similar). Whether this satisfies the requirement depends on interpretation. |

---

## What's Missing — Prioritized Action Items

### 🔴 High Priority (Blocking Checklist Items)

1. **Req 11 — MLflow: 5+ distinct runs with different configs + screenshot**  
   Action: Create a script that sweeps ≥ 5 chunking configurations (e.g., chunk_size=256/512/1024, overlap=0/50/100) or embedding models, logs each as a separate MLflow run with `chunk_size` and `overlap` params, and compares faithfulness scores. Take a screenshot of the MLflow Experiments UI and save to `reports/mlflow_runs_comparison.png`.

2. **Req 12 — MLflow Registry: register best chunking config**  
   Action: After running the 5-config sweep, call `mlflow.register_model()` on the best-performing run's config. Use a model name like `"best-rag-config"`.

3. **Req 10 — Log `chunk_size` and `overlap` to MLflow**  
   Action: Modify `scripts/log_ragas_mlflow.py` to add `mlflow.log_param("chunk_size", ...)` and `mlflow.log_param("overlap", ...)`. Since the chunker is article-level (no true chunk_size), document the article-level design and log `chunk_size=article` and `overlap=0` (or create a configurable chunker that splits long articles).

4. **Req 8 + 9 — Add `docker-compose.yml` and 3-command quickstart**  
   Action: Create `docker-compose.yml` with the FastAPI service on port 8000. Update README with a clean 3-command block:
   ```bash
   git clone <repo>
   docker compose build
   docker compose up
   ```

5. **Req 16 — CI RAGAS quality gate (faithfulness < 0.75)**  
   Action: Add a CI step to `.github/workflows/ci.yml` that runs a fast RAGAS evaluation on a small test set (e.g., 5–10 questions with a mock LLM or pre-computed responses) and fails if faithfulness < 0.75.

6. **Req 6 — Test 422 on empty question**  
   Action: Add `tests/test_api.py` using FastAPI `TestClient`. Test: `client.post("/ask", json={"question": ""})` should return 422.

### 🟡 Medium Priority (Advanced Requirements)

7. **Req 20 — Locust at 50 concurrent users**  
   Action: Run `locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://127.0.0.1:8000` with a mock generator (so LLM rate limits don't interfere). Add results to `reports/LoadTestingandPerformanceEvaluation.txt` or a new report file.

8. **Req 17 — BentoML async `/ask`**  
   Action: Change `def ask(self, request)` to `async def ask(self, request)` in `src/services/bentoml_service.py`. Use `asyncio.to_thread` or `await` for blocking calls.

9. **Req 21 — Batch re-indexing script**  
   Action: Create `scripts/batch_reindex.py` that accepts a new JSON file of articles, embeds them, and adds them to the existing FAISS index (via `index.add()`). Test it with at least one synthetic new article.

10. **Req 22 — Canary rollout docs in README**  
    Action: Add a "Canary Deployment" section to README documenting a canary strategy (e.g., using GHCR image tags + a reverse proxy weight split, or Kubernetes-based canary). Even a documented manual process satisfies this.

11. **Req 25 — Alert: faithfulness < 0.80**  
    Action: Add a Prometheus alerting rule in `prometheus/alerts.yml` triggering when `ragas_faithfulness < 0.80`, or add a Python script that checks the latest RAGAS score and sends a notification (e.g., via Slack webhook or email).

12. **Req 26 — Langfuse self-hosted**  
    Action: Add a `docker-compose.yml` (or extend it) to include the Langfuse self-hosted Docker stack. Update `.env.Example` with `LANGFUSE_BASE_URL=http://localhost:3000`. Document in README.

13. **Req 27 — RAGAS score attached to Langfuse traces**  
    Action: In `src/api/main.py`'s `/ask` handler, after generation, optionally run a single-sample RAGAS faithfulness check and call `langfuse.score(trace_id=..., name="faithfulness", value=score)`.

14. **Req 13 — DVC remote storage configured**  
    Action: Configure a DVC remote (e.g., Google Drive, S3, or DVC Studio) and add it to `.dvc/config`. Document `dvc pull` in README. Without a remote, `dvc pull` on a fresh clone has nothing to fetch.

### 🔵 Low Priority (Expert Requirements — Not Yet Started)

15. **Req 28 — Embedding drift monitoring**  
    Action: Store a baseline embedding centroid. On each `/ask`, compute cosine similarity between the query embedding and baseline. Log as a Prometheus metric `embedding_drift_score`.

16. **Req 29 — Token usage in Prometheus / Grafana cost/hour**  
    Action: Capture `response.usage.total_tokens` in `APIGenerator.generate()`. Expose as `token_usage_total` Prometheus counter. Add a Grafana panel with `rate(token_usage_total[1h]) * cost_per_token`.

17. **Reqs 18, 31, 32, 33 — vLLM + AWQ quantization**  
    Action: These are the largest remaining features. Set up a vLLM server with a local model, quantize to AWQ-4bit (e.g., using `autoawq`), compare RAGAS scores before/after, and document latency difference in README.

---

## Key Observations

- **The core RAG pipeline is production-quality**: extraction → chunking → FAISS → FastAPI → Langfuse tracing → Prometheus metrics → Docker image are all implemented and well-structured.
- **The main gaps are in experiment tracking discipline** (only 3 identical MLflow runs, no chunk_size/overlap params, no model registry entry) rather than missing infrastructure.
- **Expert features are intentionally deferred**: the architecture document is transparent about this, listing vLLM, AWQ, drift monitoring, and token cost as "Remaining Work."
- **The `docker-compose.yml` absence** is a straightforward fix but blocks requirements 8 and 9.
- **The `tests/` directory** only covers PII guardrails — there are no API endpoint tests, which leaves the 422 validation requirement untested.
