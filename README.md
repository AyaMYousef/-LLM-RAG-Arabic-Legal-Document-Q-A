Here is a concise `README.md` documenting everything built so far, serving as an onboarding and execution guide for your project repository.

---

# Arabic Legal Document Q&A (RAG) — Phase 1: Ingestion & Extraction

A production-ready RAG pipeline built for querying bilingual Arabic/English legal corpora—specifically the **Egyptian Civil Code (Law of 1948)**. This repository houses Phase 1: converting raw, unstructured, two-column PDF documents into clean, structured, and validated JSON data ready for vector database indexing.

---

## 🛠 Project Structure

```text
arabic-legal-rag/
├── data/
│   ├── raw/                  # Raw bilingual PDF (tracked by DVC)
│   └── processed/            # Structured civil_code_articles.json (tracked by DVC)
├── src/
│   ├── __init__.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── schema.py         # Pydantic v2 data models & validation
│   │   ├── normalizer.py     # Arabic text, diacritics & numeral conversion
│   │   └── pdf_parser.py     # Column-aware spatial PDF extractor
│   └── utils/
│       ├── __init__.py
│       └── logger.py         # Structured pipeline logging
├── tests/
│   ├── test_normalizer.py    # Unit tests for text & digit normalization
│   └── test_extraction.py    # Integration tests on sample pages
├── scripts/
│   └── run_pipeline.py       # Main ingestion runner
├── .gitignore
├── pyproject.toml            # PEP 621 dependencies & project metadata
└── README.md

```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites

- Python $\ge 3.10$
- Git & DVC

### 2. Environment Setup

Clone the repository, create a virtual environment, and install dependencies in editable mode:

```bash
# Clone the repository
git clone https://github.com/AyaMYousef/-LLM-RAG-Arabic-Legal-Document-Q-A.git
cd -LLM-RAG-Arabic-Legal-Document-Q-A

# Create a virtual environment
python -m venv .venv
Activate the virtual environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Windows Command Prompt:

.venv\Scripts\activate.bat

Linux / macOS:

source .venv/bin/activate
Install dependencies
pip install -e ".[dev]"

```

### 3. Data Pull (DVC)

Retrieve the raw PDF data managed by Data Version Control:

```bash
dvc pull

```

---

## ⚙️ How the Ingestion Pipeline Works

```text
[Raw PDF] ──> Spatial Layout Extraction (PyMuPDF)
          ──> Column Bounding-Box Separation (Left: EN / Right: AR)
          ──> Numeral & Text Normalization (Arabic-Indic -> Western, Diacritics Removal)
          ──> Hierarchy Tracking & Repealed Article Detection
          ──> Validation against Schema (Pydantic v2) ──> [civil_code_articles.json]

```

### Key Technical Considerations

1. **Two-Column Isolation:** Uses bounding-box coordinate slicing $(x_0, y_0, x_1, y_1)$ to prevent bilingual text interleaving.
2. **Arabic Text Normalization:** Converts Arabic-Indic numerals (`١٤٧`) to Western integers (`147`), strips Tashkeel, and standardizes Alef/Hamza forms for consistent vector embeddings.
3. **Legal Integrity:** Explicitly flags repealed articles (`is_repealed: true`) rather than deleting them, preventing vector database hallucination gaps.

---

## 🧪 Testing

Run unit tests to verify normalization and parsing logic:

```bash
pytest

```

---

## 📑 Target Data Schema

The extracted output is formatted to match the required document specification:

```json
{
  "article_number": 147,
  "book": "Obligations or Personal Rights",
  "chapter": "Sources of Obligations",
  "section": "Contracts",
  "topic": "The Effects of a Contract",
  "text_ar": "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله...",
  "text_en": "The contract makes the law of the parties...",
  "is_repealed": false,
  "source_page": 34,
  "citation": "Egyptian Civil Code, Article 147"
}
```

## Corpus Extraction

The Egyptian Civil Code PDF was processed to create a structured, article-level bilingual corpus for the legal RAG pipeline.

### Source Document

- **Input:** `data/raw/egyptian_civil_code.pdf`
- **Format:** Bilingual Arabic/English PDF
- **Length:** 170 pages
- **Target:** Extract the Civil Code article-by-article while preserving the surrounding legal hierarchy and source-page information.

### Extraction Process

The extraction pipeline:

1. Reads the PDF page by page using PyMuPDF.
2. Detects article headers from the English article numbering in the PDF.
3. Converts Arabic-Indic article numbers into normalized integer article numbers.
4. Associates each article with its surrounding:
   - Book
   - Chapter
   - Section
   - Arabic topic
   - English topic

5. Extracts the Arabic and English text belonging to each article.
6. Records the original PDF page containing the article.
7. Marks articles belonging to repealed ranges.
8. Writes the extracted corpus as structured JSON.

The resulting records follow this schema:

```text
article_number
book
chapter
section
citation
is_repealed
source_page
text_ar
text_en
topic_ar
topic_en
```

### Repealed Articles

The source document explicitly identifies the following article ranges as repealed:

- Articles **54–80**
- Articles **389–417**

These ranges are retained in the corpus with `is_repealed: true` so that the original legal numbering is preserved while allowing downstream retrieval and filtering to distinguish active provisions from repealed ones.

### Extraction Results

The latest corpus extraction produced:

- **1,149 article-level records**
- **1,094 detected English article headers**
- **1,093 expected active articles** after accounting for the repealed ranges
- Arabic and English article text stored separately where available
- Source PDF page numbers preserved for traceability

The corpus is represented at the **article level**, rather than as one large block of extracted PDF text. This structure is intended to support precise legal retrieval, citation, filtering, and later RAG evaluation.

### Validation

After extraction, a validation script was used to check the generated corpus against the expected schema and article numbering.

The validation checks include:

- Expected fields are present.
- Article numbers are normalized correctly.
- Repealed ranges are marked correctly.
- Article records can be traced back to PDF pages.
- Arabic and English text are extracted where available.
- Random article samples are inspected against the original PDF.

The validation script is run with:

```bash
uv run python scripts/validate_corpus.py
```

The validation output is also saved to:

```text
reports/corpus_validation.txt
```

The latest validation result is:

```text
RESULT: PASS
```

One source-level limitation was identified during validation:

- **Article 1022:** Arabic text is absent from the source PDF.

This is reported as a warning rather than an extraction error because the Arabic text is not present in the source document.

The resulting corpus is stored at:

```text
data/processed/corpus_raw.json
```

### DVC Versioning and Reproducibility

The source PDF and corpus extraction pipeline are tracked with DVC.

The extraction pipeline is defined in `dvc.yaml` and tracks the PDF, extraction scripts, and generated corpus as pipeline dependencies and outputs.

The extraction stage can be reproduced with:

```bash
dvc repro
```

The current DVC state can be checked with:

```bash
dvc status
```

The repository has been verified to return:

```text
Data and pipelines are up to date.
```

This confirms that the tracked source document, extraction pipeline, and generated corpus are synchronized and reproducible.

The corpus extraction and validation stage is therefore complete and ready for the next RAG pipeline stage: **article-level chunking, embedding generation, and vector-store indexing**.

### Article-Level Chunking

After corpus extraction and validation, the structured article-level corpus is converted into embedding-ready documents.

The chunking stage uses the validated corpus:

```text
data/processed/corpus_raw.json
```

Each legal article is preserved as **one logical chunk** rather than splitting Arabic and English into separate chunks. This keeps the bilingual versions of the same legal provision together and preserves article-level traceability for retrieval and citations.

For each article, the chunk contains:

* Article citation
* English topic
* Arabic topic
* English legal text
* Arabic legal text

The corresponding metadata is preserved separately:

```text
article_number
citation
source_page
book
chapter
section
topic_ar
topic_en
is_repealed
```

A simplified chunk has the following structure:

```json
{
  "text": "Egyptian Civil Code, Article 1\n\nTopic: Laws and Rights\n\nالموضوع: القانون والحق\n\nEnglish:\n...\n\nالعربية:\n...",
  "metadata": {
    "article_number": 1,
    "citation": "Egyptian Civil Code, Article 1",
    "source_page": 1,
    "book": null,
    "chapter": null,
    "section": "Laws and their Applications",
    "topic_ar": "القانون والحق",
    "topic_en": "Laws and Rights",
    "is_repealed": false
  }
}
```

The chunking implementation is located at:

```text
src/ingestion/chunker.py
```

The main functions are:

```text
build_chunk()
build_chunks()
```

`build_chunk()` converts one article into an embedding-ready document, while `build_chunks()` processes the complete corpus.

The original extracted corpus is not modified during this stage. The chunking layer provides a separate representation that can be passed to the embedding and vector-store stages.

The chunking design intentionally preserves the original article boundaries so that retrieved documents can later be mapped directly to legal citations such as:

```text
Egyptian Civil Code, Article 1
```
## Retrieval Baseline and Embedding Evaluation

After article-level chunking, the corpus was evaluated using dense semantic retrieval with FAISS.

### Embedding Models Evaluated

Two multilingual embedding models were evaluated on the same 20-question legal retrieval benchmark:

* `intfloat/multilingual-e5-base`
* `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

For E5, the recommended retrieval format was used:

* Corpus passages: `passage: <text>`
* User queries: `query: <text>`
* Embeddings were L2-normalized before indexing.
* FAISS `IndexFlatIP` was used, making inner product equivalent to cosine similarity for normalized vectors.

### Retrieval Evaluation Dataset

A fixed set of 20 Arabic legal questions was created from articles whose legal content could be directly verified in the extracted corpus.

The benchmark measures:

* **Recall@1** — expected article retrieved as the top result.
* **Recall@3** — expected article appears within the top three results.
* **Recall@5** — expected article appears within the top five results.
* **MRR (Mean Reciprocal Rank)** — measures how highly the expected article is ranked.

The same questions and expected article numbers are used for all embedding experiments to keep comparisons consistent.

### Baseline Results

| Embedding configuration  |   Recall@1 |   Recall@3 |   Recall@5 |        MRR |
| ------------------------ | ---------: | ---------: | ---------: | ---------: |
| MPNet + article metadata |     0.7500 |     0.9000 |     0.9000 |     0.8000 |
| E5 + article metadata    | **0.7500** | **0.9000** | **0.9000** | **0.8167** |
| E5 + legal text only     |     0.7500 |     0.8500 |     0.9000 |     0.8042 |

### Embedding Text Experiment

The initial chunk representation included:

```text
Article citation
English topic
Arabic topic
English legal text
Arabic legal text
```

During corpus inspection, hierarchy metadata errors were identified in some extracted records. For example, Article 802 correctly contains ownership law text, but its extracted `section` metadata incorrectly retained a heading from the preceding suretyship section.

To determine whether this metadata was negatively affecting retrieval, a second E5 experiment embedded only:

```text
Article citation
English legal text
Arabic legal text
```

while retaining `book`, `chapter`, `section`, `topic_ar`, `topic_en`, `source_page`, and `is_repealed` as metadata.

The legal-text-only configuration achieved:

* Recall@1: **0.7500**
* Recall@3: **0.8500**
* Recall@5: **0.9000**
* MRR: **0.8042**

This did not improve retrieval on the current 20-question benchmark. Therefore, the current baseline retains the original E5 chunk representation, while hierarchy metadata remains available for citation and filtering.

### Current Retrieval Baseline

The current baseline is:

```text
Embedding model:
intfloat/multilingual-e5-base

Vector store:
FAISS IndexFlatIP

Embedding dimension:
768

Corpus:
1,149 article-level chunks

Evaluation:
20-question benchmark

Recall@1:
0.7500

Recall@3:
0.9000

Recall@5:
0.9000

MRR:
0.8167
```

The retrieval benchmark is maintained as a reproducible evaluation step and can be reused when testing chunking strategies, embedding models, reranking, prompts, or other RAG components.

## Retrieval and Context Preparation

After corpus extraction and validation, the legal articles are indexed for semantic retrieval.

### 1. Article-Level Chunking

Each Civil Code article is kept as one logical chunk. The chunk contains the article citation and its English and Arabic legal text.

The chunk metadata includes:

* `article_number`
* `citation`
* `source_page`
* `book`
* `chapter`
* `section`
* `topic_ar`
* `topic_en`
* `is_repealed`

The chunking implementation is located in:

```text
src/ingestion/chunker.py
```

This preserves the article as the primary retrieval unit and avoids splitting individual legal provisions across multiple chunks.

### 2. Embedding Model

Two multilingual embedding configurations were evaluated:

* `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
* `intfloat/multilingual-e5-base`

The final retrieval baseline uses:

```text
intfloat/multilingual-e5-base
```

For E5 embeddings, the required prefixes are used:

```text
passage: <article text>
query: <user question>
```

Embeddings are normalized before being indexed.

### 3. FAISS Vector Index

The article embeddings are stored using FAISS with:

```text
IndexFlatIP
```

Because the embeddings are normalized, inner-product search corresponds to cosine similarity.

The generated E5 index is stored at:

```text
data/processed/vector_store_e5/
```

The index contains one vector per article-level chunk.

The index-building script is:

```text
scripts/build_e5_index.py
```

The vector store implementation is:

```text
src/ingestion/vector_store.py
```

### 4. Retrieval Metadata

The FAISS metadata stores both citation information and the legal text required by the downstream RAG pipeline.

Each retrieved record contains:

```text
article_number
citation
source_page
book
chapter
section
topic_ar
topic_en
is_repealed
text_en
text_ar
```

Keeping the legal text with the indexed metadata means the RAG pipeline can build context directly from retrieval results without rereading `corpus_raw.json` for every question.

### 5. Retrieval Baseline

A fixed 20-question Arabic benchmark was used to compare embedding configurations.

| Configuration            | Recall@1 | Recall@3 | Recall@5 |        MRR |
| ------------------------ | -------: | -------: | -------: | ---------: |
| MPNet + article metadata |     0.75 |     0.90 |     0.90 |     0.8000 |
| E5 + article metadata    |     0.75 |     0.90 |     0.90 | **0.8167** |
| E5 + legal text only     |     0.75 |     0.85 |     0.90 |     0.8042 |

The current baseline is:

```text
Embedding model: intfloat/multilingual-e5-base
Vector store: FAISS IndexFlatIP
Recall@1: 0.75
Recall@3: 0.90
Recall@5: 0.90
MRR: 0.8167
```

The benchmark and evaluation scripts are located in:

```text
scripts/evaluate_retrieval.py
scripts/test_e5_retrieval.py
```

### 6. Retriever

The retrieval layer is implemented in:

```text
src/rag/retriever.py
```

The retriever:

1. Receives a user question.
2. Adds the E5 `query:` prefix.
3. Generates a normalized query embedding.
4. Searches the FAISS index.
5. Returns the top-k article records with similarity scores and metadata.

For example:

```text
Question:
ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟

Top result:
Article 1
Score: 0.8436
```

Article 1 is correctly retrieved as the first result for this query.

### 7. Context Builder

Retrieved articles are converted into an LLM-ready context by:

```text
src/rag/context_builder.py
```

The context contains:

* article citation
* PDF source page
* English legal text
* Arabic legal text

Multiple retrieved articles are separated using a clear delimiter.

Example structure:

```text
Egyptian Civil Code, Article 1
PDF page: 1

English:
...

العربية:
...

---

Egyptian Civil Code, Article 200
PDF page: 24

English:
...

العربية:
...
```

This creates the boundary between semantic retrieval and answer generation.

### Current RAG Pipeline Status

The following components are now complete and tested:

```text
PDF
 ↓
Structured Article Corpus
 ↓
Article-Level Chunks
 ↓
Multilingual E5 Embeddings
 ↓
FAISS Vector Index
 ↓
Semantic Retriever
 ↓
Retrieved Legal Text
 ↓
LLM-Ready Context
```

The LLM generation layer has intentionally not yet been added. The next stage is to build the prompt layer and connect an LLM that generates a grounded answer with article-level sources.


## Vector Index and Reproducibility

After preparing and validating the structured legal corpus, the next step is to build the semantic retrieval index.

The project uses the multilingual E5 embedding model:

```text
intfloat/multilingual-e5-base
```

The corpus is converted into article-level chunks, and each article is embedded using the E5 model. The resulting embeddings are stored in a FAISS vector index together with the corresponding article metadata.

### Build the E5 Vector Index

Run:

```bash
uv run python scripts/build_e5_index.py
```

The script:

1. Loads `data/processed/corpus_raw.json`.
2. Builds article-level chunks.
3. Loads `intfloat/multilingual-e5-base`.
4. Generates normalized 768-dimensional embeddings using the `passage:` E5 prefix.
5. Builds a FAISS index.
6. Stores the index and metadata under:

```text
data/processed/vector_store_e5/
```

The generated vector store contains:

```text
data/processed/vector_store_e5/
├── index.faiss
└── metadata.json
```

### Reproducibility Across Machines

The FAISS vector store is a generated artifact and is not guaranteed to exist after cloning the repository on a new machine.

If the project is cloned onto another machine and the following file is missing:

```text
data/processed/vector_store_e5/index.faiss
```

rebuild the index using:

```bash
uv run python scripts/build_e5_index.py
```

The source of truth remains:

```text
data/processed/corpus_raw.json
```

Therefore, the vector index can be regenerated from the structured corpus whenever required.

### Retrieval Validation

The generated index is used by the RAG retriever to perform semantic search over the Egyptian Civil Code.

The retrieval pipeline has been evaluated using a set of legal questions with known expected article numbers. The current E5 baseline achieved approximately:

* Recall@1: 75%
* Recall@3: 85%
* Recall@5: 90%
* MRR: 0.80

This establishes the semantic retrieval layer before connecting an LLM for answer generation.

### Important Setup Note

The embedding model is downloaded automatically by `sentence-transformers` the first time the index is built on a new machine. A Hugging Face account/token is not required for normal model download, although unauthenticated requests may have lower rate limits.

The vector index is therefore treated as a reproducible generated artifact rather than a manually maintained dataset.


## RAG Retrieval API

After preparing the legal corpus and building the E5 vector index, the project exposes the retrieval pipeline through a FastAPI service.

### API Components

The API connects the following components:

```text
User Question
      ↓
FastAPI /ask
      ↓
Retriever
      ↓
Multilingual E5 Embeddings
      ↓
FAISS Vector Store
      ↓
Top-K Legal Articles
      ↓
Context Builder
      ↓
Article Sources
```

The current API focuses on **retrieval and source grounding**. LLM generation is intentionally not connected yet.

### Start the API

Run:

```bash
uv run uvicorn src.api.main:app --reload
```

The API is available locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available through the FastAPI Swagger UI.

### Health Check

The `GET /health` endpoint verifies that the API and vector store are available.

Example response:

```json
{
  "status": "healthy",
  "documents_indexed": 1149
}
```

This confirms that the FAISS vector store contains all 1,149 article-level records.

### Ask Endpoint

The `POST /ask` endpoint accepts a legal question:

```json
{
  "question": "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟"
}
```

The endpoint retrieves the five most relevant legal articles and returns their article numbers, citations, and source pages.

Example response:

```json
{
  "answer": "LLM generation is not connected yet. Retrieved legal context is ready for generation.",
  "sources": [
    {
      "article_number": 1,
      "citation": "Egyptian Civil Code, Article 1",
      "source_page": 1
    },
    {
      "article_number": 200,
      "citation": "Egyptian Civil Code, Article 200",
      "source_page": 24
    },
    {
      "article_number": 27,
      "citation": "Egyptian Civil Code, Article 27",
      "source_page": 4
    },
    {
      "article_number": 2,
      "citation": "Egyptian Civil Code, Article 2",
      "source_page": 1
    },
    {
      "article_number": 23,
      "citation": "Egyptian Civil Code, Article 23",
      "source_page": 4
    }
  ]
}
```

For this test question, **Article 1 is retrieved as the top result**, which is the expected legal provision.

### Current RAG Status

At this stage, the project has a working retrieval API:

* Structured bilingual Egyptian Civil Code corpus
* Article-level chunks
* Multilingual E5 embeddings
* FAISS semantic search
* Top-K retrieval
* Context construction
* Article-level source citations
* FastAPI `/health` endpoint
* FastAPI `/ask` endpoint

## Current RAG Serving and Generation Stage

The current implementation provides an end-to-end RAG question-answering pipeline:

```text
User Question
     ↓
FastAPI /ask
     ↓
E5 Query Embedding
     ↓
FAISS Vector Search
     ↓
Retrieved Legal Context
     ↓
Grounded Prompt
     ↓
API-based LLM
     ↓
Answer + Legal Sources
```

### API-based LLM

The project currently uses an OpenAI-compatible API rather than running a local generative model. This keeps the initial implementation lightweight and avoids requiring a GPU.

The generator is selected through the `.env` configuration:

```env
GENERATOR=api
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<GROQ_API_KEY>
LLM_MODEL=qwen/qwen3.8-27b
```

The application uses the same `Generator` interface for both mock and API-based generation. This allows the LLM provider to be changed later without changing the RAG pipeline.

The API generator sends:

* a system prompt containing the legal grounding rules
* a user prompt containing the retrieved legal context and question
* `temperature=0.0` for deterministic generation
* a maximum response length of 800 tokens

### Grounded Legal Generation

The system prompt explicitly restricts the model to the retrieved legal context.

The model is instructed to:

* answer only from the retrieved Egyptian Civil Code context
* avoid using outside legal knowledge
* identify the relevant article
* explicitly cite the article number
* answer Arabic questions in Arabic
* state when the retrieved context is insufficient

This is important because legal hallucinations are unacceptable for the intended use case.

### Current Retrieval Baseline

The corpus contains **1,149 indexed records** from the Egyptian Civil Code.

The current retriever uses:

```text
Embedding model: intfloat/multilingual-e5-base
Embedding dimension: 768
Vector store: FAISS
Query format: query: <question>
Top-k: 5
```

The original retrieval evaluation produced:

| Metric   | Result |
| -------- | -----: |
| Recall@1 | 0.7500 |
| Recall@3 | 0.8500 |
| Recall@5 | 0.9000 |
| MRR      | 0.8042 |

### Known Retrieval Issue

During API testing, a discrepancy was identified between the expected standalone retrieval results and the results returned through the running `/ask` endpoint.

For example, the question:

```text
ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟
```

should retrieve **Article 1**, which states the applicable order of legal sources when no applicable legislative provision exists.

The standalone retrieval previously returned Article 1 among the highest-ranked results, while the running API returned unrelated articles.

The following items have already been verified:

* the API is running from the project root
* the configured vector-store path resolves to `data/processed/vector_store_e5`
* the vector store exists
* the index contains 1,149 records
* the retriever explicitly uses `intfloat/multilingual-e5-base`

The exact cause of the API/standalone retrieval discrepancy is still under investigation.

**This issue does not block the next MLOps stages.** It will be fixed before the final retrieval/RAGAS comparison, and the evaluation will be rerun after the fix.

### Current Project Strategy

The project intentionally starts with an API-based LLM because the main objective at this stage is to implement and evaluate the MLOps pipeline rather than spend time setting up local GPU inference.

The generative serving stack will be optimized later:

```text
Current:
API-based LLM
      ↓
RAG evaluation + observability

Later:
vLLM
      ↓
AWQ-4bit quantization
      ↓
Offline/local inference
      ↓
Original vs quantized comparison
```

This separation allows evaluation, monitoring, tracing, and deployment components to be developed before introducing the additional complexity of local LLM serving.


## RAGAS Baseline Evaluation

After implementing the RAG generation pipeline, the system was evaluated using **RAGAS 0.4.3** on a dataset of **50 Arabic legal questions**.

The evaluation used the generated responses stored in:

```text
data/evaluation/ragas_responses.json
```

The evaluation was performed using the Groq-hosted LLM and the project's `intfloat/multilingual-e5-base` embedding model.

### Baseline Results

| Metric                |      Score |
| --------------------- | ---------: |
| **Faithfulness**      | **1.0000** |
| **Context Precision** | **0.8750** |
| **Context Recall**    | **1.0000** |

The detailed evaluation results are stored in:

```text
data/evaluation/ragas_results.json
```

### Interpretation

* **Faithfulness = 1.0000**
  The generated answers were fully supported by the retrieved legal context in the evaluated dataset.

* **Context Precision = 0.8750**
  Most retrieved passages were relevant to the questions, although some retrieved passages were not strictly necessary for answering the question.

* **Context Recall = 1.0000**
  The required information needed to answer the reference questions was retrieved across the evaluation dataset.

These results establish the **baseline RAG evaluation before further retrieval optimization**.

### Retrieval Optimization Note

Although the RAGAS baseline is strong, live API testing revealed an inconsistency in retrieval behavior for some questions. Therefore, the baseline should be treated as the evaluation of the stored 50-question response set rather than proof that the current live retriever is optimal.

The retrieval pipeline will be investigated and optimized separately. After retrieval improvements, the same evaluation dataset can be rerun to compare the new metrics against this baseline.

### Evaluation Flow

```text
50 Legal Questions
        ↓
RAG Retrieval
        ↓
Retrieved Legal Context
        ↓
Groq LLM Generation
        ↓
50 Generated Responses
        ↓
RAGAS 0.4.3
        ↓
┌─────────────────────────┐
│ Faithfulness    1.0000  │
│ Context Precision 0.875 │
│ Context Recall  1.0000  │
└─────────────────────────┘
        ↓
data/evaluation/ragas_results.json
```

This baseline provides a measurable reference point for subsequent retrieval optimization and MLOps monitoring.

### Retrieval Verification

Before proceeding to the next MLOps components, the retrieval pipeline was independently verified using the production E5 vector store.

For the test question:

```text
ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟
```

the active retriever loaded:

```text
Embedding model: intfloat/multilingual-e5-base
Embedding dimension: 768
Indexed documents: 1149
Vector store: data/processed/vector_store_e5
```

The top-5 retrieval results were:

| Rank | Article | Similarity |
| ---- | ------- | ---------: |
| 1    | **1**   | **0.8436** |
| 2    | 200     |     0.8309 |
| 3    | 27      |     0.8184 |
| 4    | 2       |     0.8177 |
| 5    | 23      |     0.8176 |

Article 1 is the expected legal provision for this question, confirming that the E5 embedding and FAISS retrieval pipeline can correctly identify the relevant article.

An earlier `/ask` request returned a different set of articles. Direct inspection of the `Retriever` object imported by the API confirmed that the current API configuration loads the correct E5 model and the 1,149-vector E5 index and produces the expected retrieval results. The earlier discrepancy was therefore treated as stale server state rather than an indexing or embedding problem.

## Running MLOps Tracking and Observability

This project uses **MLflow** for experiment tracking and **Langfuse** for RAG/LLM observability.

The two tools serve different purposes:

* **MLflow** tracks evaluation experiments, metrics, parameters, and artifacts.
* **Langfuse** traces individual RAG requests and shows the retrieval and LLM-generation steps.

---

### 1. Activate the Environment

From the project root:

```powershell
.venv\Scripts\activate
```

Or run commands directly through `uv` without activating the environment:

```powershell
uv run <command>
```

---

## 2. MLflow Experiment Tracking

MLflow is used to record the RAGAS evaluation results and the configuration used for the evaluation.

### Start the MLflow UI

Open a dedicated terminal in the project root:

```powershell
uv run mlflow ui
```

The local MLflow UI is available at:

```text
http://127.0.0.1:5000
```

MLflow's tracking UI allows experiments and runs to be inspected, including metrics, parameters, and artifacts.

Keep this terminal running while using the MLflow UI.

### Log the RAGAS Results

In a second terminal:

```powershell
uv run python scripts/log_ragas_mlflow.py
```

Expected output:

```text
RAGAS metrics logged to MLflow.
```

The script logs the following metrics:

```text
faithfulness
context_precision
context_recall
```

It also records parameters such as:

```text
embedding_model
llm_model
num_samples
evaluation_type
```

and stores:

```text
data/evaluation/ragas_results.json
```

as an MLflow artifact.

### View the Experiment

Open:

```text
http://127.0.0.1:5000
```

Then:

1. Open the `arabic-legal-rag` experiment.
2. Open the `ragas-baseline` run.
3. Review the logged metrics.
4. Review the parameters.
5. Open the logged RAGAS results artifact.

This provides a reproducible baseline that can later be compared against retrieval or generation improvements.

---

## 3. Langfuse Observability

Langfuse provides request-level tracing for the RAG pipeline.

The current integration uses the Langfuse Python SDK and `get_client()` with the modern observation API.

### Configure Langfuse

Create or update the local `.env` file:

```text
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Do **not** commit the `.env` file to Git.

The credentials are used by the application to send traces to Langfuse Cloud.

### Verify Langfuse Integration

Before starting the API, verify that the application and Langfuse client can be imported:

```powershell
uv run python -c "from src.api.main import app; print('API + Langfuse instrumentation OK')"
```

Expected output ends with:

```text
API + Langfuse instrumentation OK
```

### Start the RAG API

From another terminal:

```powershell
uv run uvicorn src.api.main:app
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Generate a Langfuse Trace

When an `/ask` request is processed, the application creates a root:

```text
rag-ask
```

with nested observations:

```text
rag-ask
├── retrieval
│   ├── question
│   ├── top_k
│   ├── retrieved article numbers
│   └── similarity scores
│
└── llm-generation
    ├── model
    ├── system prompt
    ├── user prompt
    └── generated answer
```

The Langfuse SDK automatically manages the lifecycle of these observations when using context managers.

### View Traces

Open the Langfuse Cloud dashboard:

```text
https://cloud.langfuse.com
```

After a request has been processed, open the corresponding trace and inspect:

* Retrieved legal articles
* Retrieval similarity scores
* LLM model
* Prompt sent to the model
* Generated answer
* End-to-end RAG execution

The application also flushes pending Langfuse events during shutdown so buffered observations are sent before the application exits.

---

## 4. Recommended Terminal Setup

For normal development, use three terminals.

### Terminal 1 — MLflow

```powershell
uv run mlflow ui
```

Keep running.

### Terminal 2 — FastAPI

```powershell
uv run uvicorn src.api.main:app
```

Keep running.

### Terminal 3 — Commands / Evaluation

Use this terminal for commands such as:

```powershell
uv run python scripts/log_ragas_mlflow.py
```

or other evaluation and testing scripts.

---

## 5. MLOps Workflow

The current MLOps workflow is:

```text
                 ┌─────────────────────┐
                 │ Egyptian Civil Code │
                 └──────────┬──────────┘
                            │
                            ▼
                    Data Processing
                            │
                            ▼
                    E5 Embeddings
                            │
                            ▼
                     FAISS Retrieval
                            │
                            ▼
                     Context Builder
                            │
                            ▼
                       LLM API
                            │
                            ▼
                         Answer
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
        Langfuse                        RAGAS
       Observability                   Evaluation
             │                             │
             │                             ▼
             │                          MLflow
             │                      Experiment Tracking
             │
             ▼
       Request Traces
```

### Responsibilities

| Tool                       | Purpose                           |
| -------------------------- | --------------------------------- |
| FastAPI                    | Serve the RAG application         |
| FAISS                      | Vector retrieval                  |
| Sentence Transformers / E5 | Multilingual embeddings           |
| Groq API                   | LLM generation                    |
| RAGAS                      | RAG quality evaluation            |
| MLflow                     | Experiment and metric tracking    |
| Langfuse                   | LLM/RAG tracing and observability |

This separation allows retrieval quality, answer quality, experiments, and production request behavior to be investigated independently.

---

## 6. Important Notes

### MLflow

MLflow is primarily used here for **offline experiment tracking**. The current baseline records the RAGAS results so that future improvements can be compared quantitatively.

### Langfuse

Langfuse is primarily used for **online/request-level observability**. It helps determine whether a problem originates from retrieval, prompt construction, or LLM generation.

### API Quotas

The LLM provider is accessed through an API. If the provider reaches its rate or token limit, `/ask` generation can temporarily fail even though retrieval and Langfuse instrumentation are functioning correctly.

### Secrets

Never commit:

```text
.env
```

or API keys such as:

```text
GROQ_API_KEY
LANGFUSE_SECRET_KEY
```

to the repository.

---

## 7. Quick Start Checklist

For a complete local MLOps session:

```powershell
# Terminal 1
uv run mlflow ui
```

```powershell
# Terminal 2
uv run uvicorn src.api.main:app
```

```powershell
# Terminal 3
uv run python scripts/log_ragas_mlflow.py
```

Then open:

```text
MLflow:
http://127.0.0.1:5000

FastAPI:
http://127.0.0.1:8000/docs

Langfuse:
https://cloud.langfuse.com
```

At this point:

* **MLflow** shows the evaluation experiment.
* **FastAPI** serves the RAG application.
* **Langfuse** shows request-level RAG traces when `/ask` requests are processed.


BentoML Serving

BentoML was added to package the existing Arabic Legal RAG pipeline as a separately deployable service.

The BentoML service reuses the existing:

Multilingual E5 embedding model
FAISS vector store
Context builder
Legal RAG prompt
Configured LLM generator

The existing FastAPI application remains available independently.

Service Architecture
                 Arabic Legal RAG
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      FastAPI                   BentoML
       :8000                     :3000
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
                E5 + FAISS
                       │
                       ▼
                Context Builder
                       │
                       ▼
                   LLM API

This separation allows the same RAG logic to be exposed through the original FastAPI application and packaged as a BentoML service.

Installation

BentoML was installed with:

uv add bentoml

The project currently uses:

BentoML 1.4.39
Service Location

The BentoML service is located at:

src/services/bentoml_service.py

The service is named:

LegalRAGService
Start the BentoML Service

From the project root:

uv run bentoml serve src.services.bentoml_service:LegalRAGService

The service runs on:

http://127.0.0.1:3000
Health Check

BentoML exposes a health endpoint that can be checked with PowerShell:

Invoke-WebRequest http://127.0.0.1:3000/healthz -UseBasicParsing

A successful response returns:

StatusCode : 200
StatusDescription : OK

This confirms that the BentoML service has initialized successfully.


## Prometheus Monitoring

The RAG API exposes application and RAG-specific metrics using `prometheus-fastapi-instrumentator` and the Prometheus Python client.

### Metrics

The following custom metrics are exposed at `/metrics`:

* `rag_requests_total` — total number of RAG requests.
* `rag_errors_total` — total number of failed RAG requests.
* `rag_retrieval_latency_seconds` — retrieval latency histogram.
* `rag_llm_latency_seconds` — LLM generation latency histogram.

FastAPI HTTP metrics and Python runtime/GC metrics are also exposed.

### Verify the Metrics Endpoint

Start the API:

```bash
uv run uvicorn src.api.main:app --reload
```

Verify the metrics endpoint:

```bash
curl http://127.0.0.1:8000/metrics
```

The response should contain metrics such as:

```text
rag_requests_total
rag_errors_total
rag_retrieval_latency_seconds
rag_llm_latency_seconds
```

### Prometheus Configuration

The project contains:

```text
prometheus/
└── prometheus.yml
```

Configuration:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "arabic-legal-rag"
    metrics_path: "/metrics"
    static_configs:
      - targets:
          - "host.docker.internal:8000"
```

### Run Prometheus with Docker

Docker Desktop is required.

Pull the Prometheus image:

```bash
docker pull prom/prometheus
```

Start Prometheus:

```bash
docker run -d \
  --name arabic-legal-prometheus \
  -p 9090:9090 \
  -v "$(pwd)/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

Verify that the container is running:

```bash
docker ps
```

Prometheus is available at:

```text
http://localhost:9090
```

### Verify the Prometheus Target

Open:

```text
http://localhost:9090
```

Then navigate to:

**Status → Targets**

The `arabic-legal-rag` target should show:

```text
UP
```

A successful target means Prometheus is successfully scraping:

```text
http://host.docker.internal:8000/metrics
```

### Example PromQL Queries

Check whether the API target is available:

```promql
up
```

Check total RAG requests:

```promql
rag_requests_total
```

Check retrieval observations:

```promql
rag_retrieval_latency_seconds_count
```

Check LLM generation observations:

```promql
rag_llm_latency_seconds_count
```

At startup, the RAG counters remain at `0` until `/ask` requests are processed.

### Prometheus Architecture

```text
┌─────────────────────┐
│     FastAPI RAG     │
│       :8000         │
└──────────┬──────────┘
           │
        /metrics
           │
           ▼
┌─────────────────────┐
│     Prometheus      │
│       :9090         │
│      Docker         │
└─────────────────────┘
```

---

## Grafana Monitoring

Grafana is used to visualize the Prometheus metrics exposed by the Arabic Legal RAG API.

### Run Grafana with Docker

Docker Desktop must be running.

Pull the Grafana image:

```bash
docker pull grafana/grafana
```

Start Grafana:

```bash
docker run -d \
  --name arabic-legal-grafana \
  -p 3000:3000 \
  grafana/grafana
```

Verify that both monitoring containers are running:

```bash
docker ps
```

Expected containers:

```text
arabic-legal-prometheus
arabic-legal-grafana
```

Grafana is available at:

```text
http://localhost:3000
```

### Grafana Initial Login

The default credentials are:

```text
Username: admin
Password: admin
```

Grafana may request a password change during the first login.

### Configure Prometheus as a Data Source

In Grafana:

1. Open **Connections → Data sources**.
2. Select **Add new data source**.
3. Select **Prometheus**.
4. Set the Prometheus server URL to:

```text
http://host.docker.internal:9090
```

Because Grafana runs inside Docker, `localhost:9090` refers to the Grafana container itself. `host.docker.internal` allows the Grafana container to reach Prometheus through the host.

Click **Save & test**.

Successful configuration displays:

```text
Successfully queried the Prometheus API.
```

### RAG Monitoring Dashboard

Dashboard name:

**Arabic Legal RAG — Production Monitoring**

The dashboard contains six monitoring panels.

#### 1. RAG Request Rate

Shows the rate of incoming RAG requests over the previous five minutes.

```promql
rate(rag_requests_total[5m])
```

#### 2. RAG Error Rate

Shows the rate of failed RAG requests.

```promql
rate(rag_errors_total[5m])
```

#### 3. Retrieval Latency — p95

Shows the 95th percentile document retrieval latency.

```promql
histogram_quantile(
  0.95,
  rate(rag_retrieval_latency_seconds_bucket[5m])
)
```

#### 4. LLM Generation Latency — p95

Shows the 95th percentile LLM generation latency.

```promql
histogram_quantile(
  0.95,
  rate(rag_llm_latency_seconds_bucket[5m])
)
```

#### 5. API Request Latency — p95

Shows the 95th percentile FastAPI request latency.

```promql
histogram_quantile(
  0.95,
  rate(http_request_duration_seconds_bucket[5m])
)
```

#### 6. RAG API Availability

Displays whether the Prometheus target is currently available.

```promql
up{job="arabic-legal-rag"}
```

Values:

```text
1 = UP
0 = DOWN
```

### Monitoring Architecture

```text
                    ┌─────────────────────┐
                    │     FastAPI RAG      │
                    │       :8000          │
                    └──────────┬──────────┘
                               │
                            /metrics
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Prometheus      │
                    │       :9090         │
                    │      Docker         │
                    └──────────┬──────────┘
                               │
                           PromQL queries
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Grafana       │
                    │       :3000         │
                    │      Docker         │
                    └─────────────────────┘
```

### Validation

The monitoring stack was validated end-to-end:

* FastAPI exposes `/metrics`.
* Custom RAG Prometheus metrics are registered.
* Prometheus runs successfully in Docker.
* Prometheus successfully scrapes the FastAPI metrics endpoint.
* The `arabic-legal-rag` Prometheus target reports **UP**.
* Grafana runs successfully in Docker.
* Grafana successfully queries Prometheus.
* The **Arabic Legal RAG — Production Monitoring** dashboard has been configured.

At the time of monitoring setup validation, no `/ask` requests had been processed after the Prometheus instrumentation was enabled, so the RAG request and latency metrics had not yet accumulated observations.
