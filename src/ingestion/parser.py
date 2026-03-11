"""Document parser — converts raw PDF/DOCX documents to clean Markdown.

Uses the Docling library for document conversion with optional OCR (EasyOCR)
and VLM (Azure OpenAI or local SmolVLM) support.  The pipeline is:

1. **Convert** — Docling parses the raw document into a structured model.
2. **Filter**  — Remove noise elements (headers, footers, TOC, logos, etc.).
3. **Describe** — If VLM is enabled, generate text descriptions of figures.
4. **Export**  — Convert the cleaned document model to Markdown.
5. **Clean**   — Post-process the Markdown string (collapse whitespace, etc.).
6. **Write**   — Save the final Markdown to ``cache/markdown/{doc_id}.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    TableStructureOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_model_specs import SMOLDOCLING_TRANSFORMERS
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc.document import DoclingDocument, NodeItem, PictureItem

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────


@dataclass(frozen=True)
class ParserConfig:
    """All knobs for the document parsing pipeline.

    Sensible defaults are provided for every field.  Override via CLI
    flags (``--ocr``, ``--vlm``, etc.) or by constructing directly.
    """

    # ── OCR ─────────────────────────────────────────────────────
    ocr_enabled: bool = False
    """Activate EasyOCR for scanned / image-heavy pages."""

    # ── VLM (vision-language model) ─────────────────────────────
    vlm_enabled: bool = False
    """Activate a vision-language model for layout understanding
    and image description."""

    vlm_backend: Literal["azure", "local"] = "azure"
    """Which VLM backend to use.
    - ``"azure"``  — Azure OpenAI endpoint (requires credentials).
    - ``"local"``  — Docling's built-in SmolVLM (runs on-device).
    """

    # ── Azure OpenAI credentials (only used when vlm_backend="azure") ──
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_model: str = "gpt-4.1"
    azure_api_version: str = "2023-05-15"

    # ── Table extraction ────────────────────────────────────────
    table_mode: Literal["accurate", "fast"] = "accurate"
    """Docling ``TableFormerMode``: ``"accurate"`` is slower but better;
    ``"fast"`` is lighter on resources."""

    # ── Post-processing / cleanup toggles ───────────────────────
    strip_headers_footers: bool = True
    """Remove elements classified as page headers or footers."""

    strip_toc: bool = True
    """Remove table-of-contents sections."""

    strip_logos_icons: bool = True
    """Remove images classified as logos or icons."""

    strip_page_numbers: bool = True
    """Remove standalone page-number elements."""

    describe_images: bool = True
    """When VLM is enabled, generate a text description for
    content-bearing figures before stripping the image data.
    Has no effect when ``vlm_enabled`` is False."""


# ── Parser output ───────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedDocument:
    """Result of successfully parsing a single document.

    Attributes
    ----------
    doc_id:
        SHA-256 hex digest of the original file (from the registry).
    source_path:
        Absolute path to the original document that was parsed.
    markdown:
        Cleaned Markdown content ready for chunking.
    title:
        Document title extracted from metadata or the first heading.
    page_count:
        Number of pages in the source document.
    metadata:
        Any additional metadata extracted during conversion
        (author, date, language, etc.).
    """

    doc_id: str
    source_path: Path
    markdown: str
    title: str
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Converter builder ───────────────────────────────────────────

# Prompt used when VLM describes content-bearing figures.
_IMAGE_DESCRIPTION_PROMPT = (
    "Describe the content of this technical diagram or figure in 2-3 sentences. "
    "Focus on what information it conveys, not its visual style."
)


def _build_converter(config: ParserConfig) -> DocumentConverter:
    """Construct a Docling ``DocumentConverter`` configured per *config*.

    Two mutually exclusive pipeline modes are supported:

    * **Standard PDF pipeline** (default) — layout analysis + OCR + table
      extraction.  Used when ``vlm_enabled`` is False.
    * **VLM pipeline** — uses a vision-language model for full-page
      conversion.  Used when ``vlm_enabled`` is True.

    DOCX always uses Docling's simple pipeline (no special options needed).
    """

    if config.vlm_enabled:
        return _build_vlm_converter(config)
    return _build_standard_converter(config)


def _build_standard_converter(config: ParserConfig) -> DocumentConverter:
    """Build a converter using the standard PDF pipeline (no VLM)."""

    # ── Table structure ─────────────────────────────────────────
    table_mode = (
        TableFormerMode.ACCURATE
        if config.table_mode == "accurate"
        else TableFormerMode.FAST
    )
    table_options = TableStructureOptions(mode=table_mode)

    # ── OCR ─────────────────────────────────────────────────────
    ocr_options: OcrAutoOptions | EasyOcrOptions = (
        EasyOcrOptions(force_full_page_ocr=False)
        if config.ocr_enabled
        else OcrAutoOptions()
    )

    # ── Assemble PDF pipeline options ───────────────────────────
    pdf_pipeline_opts = PdfPipelineOptions(
        do_ocr=config.ocr_enabled,
        do_table_structure=True,
        table_structure_options=table_options,
        ocr_options=ocr_options,
        do_picture_classification=True,
        do_picture_description=False,
        generate_picture_images=False,
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_opts,
                pipeline_cls=StandardPdfPipeline,
            ),
            InputFormat.DOCX: WordFormatOption(),
        },
    )


def _build_vlm_converter(config: ParserConfig) -> DocumentConverter:
    """Build a converter using the VLM pipeline for full-page understanding."""

    if config.vlm_backend == "local":
        # Use Docling's built-in SmolDocling model (runs on-device).
        vlm_options = SMOLDOCLING_TRANSFORMERS
    else:
        # Use an Azure OpenAI endpoint as the VLM backend.
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        from pydantic import AnyUrl

        if not config.azure_endpoint or not config.azure_api_key:
            raise ValueError(
                "Azure VLM backend requires 'azure_endpoint' and "
                "'azure_api_key' in ParserConfig."
            )

        # Build the chat-completions URL from the Azure endpoint.
        base = config.azure_endpoint.rstrip("/")
        url = (
            f"{base}/openai/deployments/{config.azure_model}"
            f"/chat/completions?api-version={config.azure_api_version}"
        )

        vlm_options = ApiVlmOptions(  # type: ignore[call-arg]
            url=AnyUrl(url),
            headers={"api-key": config.azure_api_key},
            prompt="Convert this page to docling.",
            response_format=ResponseFormat.DOCTAGS,
            timeout=120.0,
        )

    # ── Picture description (VLM-describe-then-strip) ──────────
    pic_desc_options: PictureDescriptionApiOptions | PictureDescriptionVlmOptions
    do_picture_desc = config.describe_images

    if config.describe_images:
        if config.vlm_backend == "local":
            pic_desc_options = PictureDescriptionVlmOptions(
                repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
                prompt=_IMAGE_DESCRIPTION_PROMPT,
            )
        else:
            base = config.azure_endpoint.rstrip("/")  # type: ignore[union-attr]
            pic_url = (
                f"{base}/openai/deployments/{config.azure_model}"
                f"/chat/completions?api-version={config.azure_api_version}"
            )
            from pydantic import AnyUrl as _AnyUrl

            pic_desc_options = PictureDescriptionApiOptions(
                url=_AnyUrl(pic_url),
                headers={"api-key": config.azure_api_key},  # type: ignore[arg-type]
                prompt=_IMAGE_DESCRIPTION_PROMPT,
                timeout=60.0,
            )
    else:
        # Provide a harmless default (won't be used since do_picture_description=False).
        pic_desc_options = PictureDescriptionVlmOptions(
            repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
            prompt=_IMAGE_DESCRIPTION_PROMPT,
        )

    # ── Assemble VLM pipeline options ───────────────────────────
    vlm_pipeline_opts = VlmPipelineOptions(
        vlm_options=vlm_options,
        do_picture_classification=True,
        do_picture_description=do_picture_desc,
        generate_page_images=True,
        generate_picture_images=config.describe_images,
        picture_description_options=pic_desc_options,
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=vlm_pipeline_opts,
                pipeline_cls=VlmPipeline,
            ),
            InputFormat.DOCX: WordFormatOption(),
        },
    )


# ── Pre-export element filtering ────────────────────────────────

# Labels that correspond to logos, icons, and other non-content images.
_LOGO_ICON_CLASSES: frozenset[str] = frozenset({
    "LOGO", "ICON", "STAMP", "QR_CODE", "BAR_CODE", "SIGNATURE",
})


def _filter_document_elements(
    doc: DoclingDocument,
    config: ParserConfig,
) -> DoclingDocument:
    """Remove noise elements from a Docling document **in-place**.

    Walks the document tree and collects elements to delete based on
    their ``DocItemLabel`` and (for pictures) their classification
    annotations.  All matching elements are removed in a single
    ``delete_items`` call at the end to avoid iterator invalidation.

    For pictures that are NOT classified as logos/icons (i.e. real
    diagrams or charts), if the VLM produced a ``DescriptionAnnotation``
    we keep the description text by inserting a ``[Figure: …]`` text
    element before deleting the picture.

    Returns the same ``doc`` reference (mutated).
    """
    from docling_core.types.doc.document import (
        DescriptionAnnotation,
        DocItem,
        PictureClassificationData,
        PictureItem,
        TextItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    items_to_delete: list[NodeItem] = []
