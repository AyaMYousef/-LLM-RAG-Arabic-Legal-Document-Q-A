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

* **Input:** `data/raw/egyptian_civil_code.pdf`
* **Format:** Bilingual Arabic/English PDF
* **Length:** 170 pages
* **Target:** Extract the Civil Code article-by-article while preserving the surrounding legal hierarchy and source-page information.

### Extraction Process

The extraction pipeline:

1. Reads the PDF page by page using PyMuPDF.
2. Detects article headers from the English article numbering in the PDF.
3. Converts Arabic-Indic article numbers into normalized integer article numbers.
4. Associates each article with its surrounding:

   * Book
   * Chapter
   * Section
   * Arabic topic
   * English topic
   
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

* Articles **54–80**
* Articles **389–417**

These ranges are retained in the corpus with `is_repealed: true` so that the original legal numbering is preserved while allowing downstream retrieval and filtering to distinguish active provisions from repealed ones.

### Extraction Results

The latest corpus extraction produced:

* **1,149 article records**
* **1,094 detected English article headers**
* **1,093 expected active articles** after accounting for the repealed ranges
* Arabic and English article text are stored separately
* Source PDF page numbers are preserved for traceability

The corpus is therefore represented at the **article level**, rather than as one large block of extracted PDF text. This structure is intended to support precise legal retrieval, citation, filtering, and later RAG evaluation.

### Validation

After extraction, a validation script was used to check the generated corpus against the expected schema and article numbering.

The validation checks include:

* Expected fields are present.
* Article numbers are normalized correctly.
* Repealed ranges are marked correctly.
* Article records can be traced back to PDF pages.
* Arabic and English text are extracted where available.
* Random article samples are inspected against the original PDF.

The validation report is generated with:

```bash
uv run python scripts/validate_corpus.py
```

To save the validation output as a text file:

```bash
uv run python scripts/validate_corpus.py | tee reports/validation_report.txt
```

The resulting corpus is stored at:

```text
data/processed/corpus_raw.json
```

This extraction and validation stage is completed before introducing DVC versioning for the dataset and pipeline artifacts.
