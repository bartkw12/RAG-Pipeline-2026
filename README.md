# UC37.5 RAG Pipeline

A **Retrieval-Augmented Generation** pipeline for engineering and technical documents.

## Project structure

- `src/ingestion/` — document ingestion, parsing, chunking, embedding
- `src/retrieval/` — search and similarity matching
- `src/generation/` — response generation
- `src/config/` — centralized paths, environment loading, and configuration
- `src/cli/` — command-line interface
- `input/` — user-provided documents for ingestion (drop folder)
- `cache/` — derived artifacts (markdown, chunks, embeddings, metadata)

## Quick start

1. Clone the repository.
2. Copy `.env.example` to `.env`.
3. Add your own API keys and endpoints in `.env`.
4. Run pipeline modules from `src/` as they are added.

## Document ingestion

The ingestion pipeline processes engineering documents (PDF, DOCX) into normalized, indexed formats ready for retrieval.

### Three ways to select documents

Documents are selected for ingestion using one of three methods (in order of priority):

#### 1. CLI paths/globs (highest priority)

Explicitly specify file paths or glob patterns on the command line:

```bash
python -m src.cli.ingest --paths "D:\Specs\*.pdf" report.docx
python -m src.cli.ingest --paths "./input/project_a/**/*.pdf"
```

When `--paths` is used, the input folder and any manifest are ignored.

#### 2. Manifest file (second priority)

Create a `manifest.json` (or any path) with include/exclude rules:

```bash
python -m src.cli.ingest --manifest manifest.json
```

Manifest format:

```json
{
  "roots": [
    "./input",
    "D:/Engineering/Archive"
  ],
  "include": [
    "**/*.pdf",
    "**/*.docx"
  ],
  "exclude": [
    "**/~$*.docx",
    "**/draft_*"
  ],
  "files": [
    "C:/Special/critical-spec.pdf"
  ]
}
```

- `roots` (optional): directories to search. Defaults to `input/`.
- `include` (optional): glob patterns to include. Defaults to all supported extensions.
- `exclude` (optional): patterns to skip.
- `files` (optional): explicit file paths to always include.

See [manifest.example.json](manifest.example.json) for a complete example.

#### 3. Drop folder (default)

Simply drop files into the `input/` directory and run:

```bash
python -m src.cli.ingest
```

The pipeline recursively scans `input/` for all supported file types. Already-processed files in `input/processed/` are skipped.

### Duplicate detection and tracking

Every file is identified by a **SHA-256 hash of its content**. This means:

- **Same content = same document**: if you submit the identical file from different locations or with a different name, it is recognized as a duplicate and skipped.
- **Updated content = re-ingest**: if a file was previously ingested but its content changed, it is automatically re-ingested and the old entry replaced.
- **No re-processing of unchanged files**: identical files are never parsed twice, saving time and ensuring consistency.

The ingestion log reports exactly what happened:

```
  ⓘ Skipped (unchanged): 'spec.pdf' — identical content was already ingested as 'spec.pdf' on 2026-03-10T14:30:00+00:00.
  ⓘ Re-ingesting (modified): 'report.docx' — content changed since last ingestion on 2026-03-10T10:15:00+00:00.
  ⓘ New file: 'architecture.pdf' — queued for ingestion.
```

Metadata is stored in `cache/meta/ingestion_registry.json` and persists across runs.

### Document parsing

Documents are converted to clean Markdown using the [Docling](https://github.com/DS4SD/docling) library. The parser supports PDF and DOCX files out of the box.

#### Default mode (standard pipeline)

By default, parsing uses Docling's standard PDF pipeline — no OCR, no VLM. This is fast and works well for text-based PDFs and all DOCX files:

```bash
python -m src.cli.ingest --paths report.pdf
```

#### OCR (EasyOCR)

For scanned documents or image-heavy pages, enable OCR:

```bash
python -m src.cli.ingest --paths scanned.pdf --ocr
```

This activates EasyOCR via Docling's pipeline. Text from scanned pages is extracted and included in the Markdown output.

#### VLM (Vision-Language Model) pipeline

For richer extraction — including table understanding and diagram descriptions — enable the VLM pipeline:

```bash
# Azure OpenAI backend (default)
python -m src.cli.ingest --paths complex.pdf --vlm

# Local SmolDocling backend
python -m src.cli.ingest --paths complex.pdf --vlm --vlm-backend local
```

When VLM is enabled:
- **Azure backend** (`--vlm-backend azure`, default): Uses your Azure OpenAI credentials from `config_V2025_05_31.json`. Requires an endpoint with a vision-capable model (e.g. GPT-4.1).
- **Local backend** (`--vlm-backend local`): Uses the SmolDocling Transformers model locally. No API key needed, but requires GPU for reasonable performance.
- **Image descriptions**: Content-bearing figures and diagrams are described in text by the VLM before images are stripped. The prompt focuses on *what information the figure conveys*, not its visual style. Descriptions appear as `[Figure: ...]` blocks in the Markdown.

#### Table extraction mode

Choose between accurate (default) and fast table extraction:

```bash
# Accurate mode (TableFormer — slower, better quality)
python -m src.cli.ingest --paths tables.pdf --table-mode accurate

# Fast mode (quicker, less precise)
python -m src.cli.ingest --paths tables.pdf --table-mode fast
```

#### Post-processing and filtering

By default, the parser automatically strips:
- Page headers and footers
- Table-of-contents sections
- Logos and icons
- Residual page-number lines (e.g. "Page 3 of 12")
- Excess blank lines and trailing whitespace

You can selectively disable this filtering:

```bash
# Keep page headers and footers
python -m src.cli.ingest --paths doc.pdf --no-strip-headers

# Keep table-of-contents
python -m src.cli.ingest --paths doc.pdf --no-strip-toc

# Keep image placeholders (skip image stripping)
python -m src.cli.ingest --paths doc.pdf --keep-images
```

#### Combining parsing flags

All parsing flags can be combined with each other and with selection/behaviour flags:

```bash
# Full pipeline: OCR + VLM + Azure, force re-ingest, verbose
python -m src.cli.ingest --paths "D:\Specs\*.pdf" --ocr --vlm --force --verbose

# Drop-folder mode with fast tables and no header stripping
python -m src.cli.ingest --table-mode fast --no-strip-headers

# Dry run to preview what VLM parsing would process
python -m src.cli.ingest --vlm --dry-run
```

#### Output

Parsed Markdown files are written to `cache/markdown/{doc_id}.md`, where `doc_id` is the SHA-256 hash of the source file. Each `ParsedDocument` also captures:
- Document title (from metadata or first heading)
- Page count
- Source format, conversion status, and file size

### CLI reference

```bash
# Default: scan input/ folder
python -m src.cli.ingest

# Explicit paths (highest priority)
python -m src.cli.ingest --paths "D:\Specs\*.pdf" report.docx

# Manifest file (second priority)
python -m src.cli.ingest --manifest manifest.json

# Preview without doing anything
python -m src.cli.ingest --dry-run

# Force re-ingest all, bypass duplicate detection
python -m src.cli.ingest --force

# Enable OCR for scanned documents
python -m src.cli.ingest --ocr

# Enable VLM pipeline (Azure backend by default)
python -m src.cli.ingest --vlm

# Enable VLM with local SmolDocling model
python -m src.cli.ingest --vlm --vlm-backend local

# Choose table extraction mode
python -m src.cli.ingest --table-mode fast

# Keep headers/footers, TOC, or images
python -m src.cli.ingest --no-strip-headers --no-strip-toc --keep-images

# Verbose logging (DEBUG level)
python -m src.cli.ingest --verbose

# Quiet mode (errors only; summary still prints)
python -m src.cli.ingest --quiet

# Override the default input directory
python -m src.cli.ingest --input-dir "D:\Engineering\Docs"

# View help
python -m src.cli.ingest --help
```

### File handling

- **Drop-folder files** (`input/`): After successful ingestion, moved to `input/processed/` with subdirectory structure preserved. Safe to delete later.
- **CLI/manifest files** (external locations): Never moved or modified. Remain in their original locations.

### Exit codes

- `0` — success (files ingested or skipped as expected)
- `1` — no files found to ingest
- `2` — parsing failures occurred

Useful for scripting and CI/CD pipelines.

## Security model

- Real secrets stay in `.env` (ignored by git).
- `.env.example` is committed as a template only.
- `config*.json` and common private key/cert files are ignored.

## Configuration

### Environment variables

All paths and settings can be overridden via environment variables. Copy `.env.example` to `.env` and customize:

```dotenv
# (Optional) Override default paths
RAG_INPUT_DIR=./input
RAG_CACHE_DIR=./cache

# Your API keys and endpoints
API_BASE_URL=https://api.example.com
API_KEY=<your-key-here>
```

Use the helpers in `src/config/` to load and validate:

```python
from config import load_local_env, require_env, ensure_dirs

# Load local .env file
load_local_env()

# Validate required keys are set
config = require_env(["API_KEY", "DB_URL"])

# Create all required directories
ensure_dirs()
```

### Path constants

Import centralized path constants from `src/config/`:

```python
from config import (
    PROJECT_ROOT,
    INPUT_DIR,
    PROCESSED_DIR,
    CACHE_DIR,
    MARKDOWN_DIR,
    CHUNK_DIR,
    EMBED_DIR,
    META_DIR,
    REGISTRY_FILE,
    MANIFEST_DEFAULT,
    SUPPORTED_EXTENSIONS,
)
```

This ensures all modules use the same paths and respects env-var overrides.
