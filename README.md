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

* Python $\ge 3.10$
* Git & DVC

### 2. Environment Setup

Clone the repository, create a virtual environment, and install dependencies in editable mode:

```bash
# Clone the repository
git clone <https://github.com/AyaMYousef/-LLM-RAG-Arabic-Legal-Document-Q-A.git>
cd arabic-legal-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core and dev dependencies
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