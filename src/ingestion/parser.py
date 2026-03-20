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

import html as html_module
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from PIL import Image

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
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
    """Activate Azure GPT-4.1 vision analysis for figures and
    diagrams detected by the standard pipeline."""

    # ── Azure OpenAI credentials (used when vlm_enabled=True) ──
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

    strip_boilerplate_blocks: bool = True
    """Remove repeated boilerplate content that Docling fails to
    classify as page headers — e.g. security classification banners
    (``THALES GROUP INTERNAL``) and repeated document-ID tables."""

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

# System prompt for GPT-4.1 VLM figure descriptions.
_VLM_SYSTEM_PROMPT = (
    "You are analyzing a figure from a technical engineering document. "
    "In 2-4 sentences, state: (1) what type of figure this is "
    "(e.g., circuit schematic, block diagram, architecture diagram, "
    "oscilloscope capture, flowchart, reference drawing, test setup "
    "photo, timing diagram, plot/graph, pin-out diagram, wiring diagram); "
    "(2) the specific measurements, values, labels, or parameters "
    "visible in the image (e.g., component IDs and values, signal names, "
    "voltage levels, pin labels, connector references, thresholds, "
    "frequencies, axis values, decision nodes, module names); "
    "(3) the key technical finding or relationship shown. "
    "Read and report actual values, component designators, and labels "
    "from the image. "
    "Do not give generic definitions or explain general concepts. "
    "Be precise and concise."
)


def _build_converter(config: ParserConfig) -> DocumentConverter:
    """Construct a Docling ``DocumentConverter`` configured per *config*.

    Uses the standard PDF pipeline for layout analysis, OCR, and table
    extraction.  Picture descriptions are handled separately by
    ``_describe_pictures()`` after conversion.
    """

    return _build_standard_converter(config)


def _build_standard_converter(config: ParserConfig) -> DocumentConverter:
    """Build a converter using the standard PDF pipeline.

    Picture classification is enabled so the filter can distinguish
    logos/icons from content figures.  VLM descriptions are done in
    a separate post-conversion step.
    """

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
    # Note: do_picture_description=False — we handle VLM separately
    # to avoid Pillow crashes and to inject figure caption context.
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


# ── VLM figure description (Azure GPT-4.1) ─────────────────────


def _resolve_ref(doc: DoclingDocument, ref) -> object | None:
    """Resolve a ``RefItem`` (JSON pointer like ``#/texts/42``) on *doc*."""
    cref = getattr(ref, "cref", None)
    if not cref or not isinstance(cref, str):
        return None
    parts = cref.lstrip("#/").split("/")
    if len(parts) != 2:
        return None
    collection_name, index_str = parts
    collection = getattr(doc, collection_name, None)
    if collection is None or not isinstance(collection, list):
        return None
    try:
        return collection[int(index_str)]
    except (ValueError, IndexError):
        return None


def _get_caption_text(doc: DoclingDocument, item: PictureItem) -> str:
    """Extract the caption/title text for a ``PictureItem``."""
    parts: list[str] = []
    for ref in item.captions or []:
        ref_item = _resolve_ref(doc, ref)
        text = getattr(ref_item, "text", "") or ""
        if text.strip():
            parts.append(text.strip())
    return " ".join(parts)


_RE_FIGURE_NUMBER = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)

_RE_FIGURE_PREFIX = re.compile(
    r"^(?:Figure|Fig\.?)\s*\d+\s*[:.\-]\s*", re.IGNORECASE
)


def _extract_figure_number(caption: str) -> str | None:
    """Extract figure number from a caption like 'Figure 15: ...'."""
    m = _RE_FIGURE_NUMBER.search(caption)
    return m.group(1) if m else None


def _caption_without_prefix(caption: str) -> str:
    """Strip the 'Figure N:' prefix from a caption if present."""
    return _RE_FIGURE_PREFIX.sub("", caption).strip()


def _call_azure_vlm(
    image_bytes: bytes,
    prompt: str,
    config: ParserConfig,
) -> str | None:
    """Send an image to Azure GPT-4.1 and return the description text."""
    import base64

    import requests

    base = (config.azure_endpoint or "").rstrip("/")
    url = (
        f"{base}/openai/deployments/{config.azure_model}"
        f"/chat/completions?api-version={config.azure_api_version}"
    )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            url,
            headers={"api-key": config.azure_api_key},
            json=payload,
            timeout=60,
        )
        if not resp.ok:
            logger.warning(
                "Azure VLM API returned %s: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Azure VLM API call failed.")
        return None


def _crop_picture_image(
    item: PictureItem,
    pdf_pages: dict[int, Image.Image],
) -> bytes | None:
    """Crop a picture from a pre-rendered page image and return PNG bytes.

    *pdf_pages* maps 1-based page numbers to rendered PIL images.
    Returns None on any error (corrupt crop, missing page, etc.).
    """
    from io import BytesIO

    try:
        if not item.prov:
            return None

        prov = item.prov[0]
        page_no = prov.page_no
        page_img = pdf_pages.get(page_no)
        if page_img is None:
            return None

        bbox = prov.bbox
        # Docling bbox: l, t, r, b in points (72 dpi).
        # Page rendered at scale=5.0 → 360 dpi.
        scale = 5.0
        left = int(bbox.l * scale)
        top = int(bbox.t * scale)
        right = int(bbox.r * scale)
        bottom = int(bbox.b * scale)

        # Clamp to page bounds
        pw, ph = page_img.size
        left = max(0, min(left, pw))
        top = max(0, min(top, ph))
        right = max(left + 1, min(right, pw))
        bottom = max(top + 1, min(bottom, ph))

        cropped = page_img.crop((left, top, right, bottom))
        buf = BytesIO()
        cropped.save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        logger.debug(
            "Failed to crop picture at page %s: %s",
            getattr(item.prov[0], "page_no", "?") if item.prov else "?",
            item.self_ref,
            exc_info=True,
        )
        return None


def _render_pdf_pages(
    pdf_path: Path,
    page_numbers: set[int],
    scale: int = 5,
) -> dict[int, Image.Image]:
    """Render specific PDF pages as PIL images using pypdfium2.

    *page_numbers* are 1-based.  Returns a dict mapping page number
    to its rendered PIL image.
    """
    import pypdfium2

    pages: dict[int, Image.Image] = {}
    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        for page_no in sorted(page_numbers):
            idx = page_no - 1  # pypdfium2 uses 0-based indexing
            if 0 <= idx < len(pdf):
                try:
                    page = pdf[idx]
                    bitmap = page.render(scale=scale)
                    pages[page_no] = bitmap.to_pil()
                except Exception:
                    logger.debug(
                        "Failed to render page %d from '%s'.",
                        page_no, pdf_path.name, exc_info=True,
                    )
        pdf.close()
    except Exception:
        logger.warning(
            "Could not open PDF '%s' for image rendering.", pdf_path.name,
            exc_info=True,
        )
    return pages


def _describe_pictures(
    doc: DoclingDocument,
    pdf_path: Path,
    config: ParserConfig,
) -> dict[str, str]:
    """Call Azure GPT-4.1 for each content-bearing picture.

    Returns a dict mapping ``item.self_ref`` → formatted description
    string like ``[VLM - Figure 3: Test Set-up] This is a ...``.

    Skips logos/icons and pictures that fail to crop.
    """
    if not (config.vlm_enabled and config.describe_images):
        return {}

    if not config.azure_endpoint or not config.azure_api_key:
        logger.warning(
            "VLM enabled but Azure credentials missing — skipping picture descriptions."
        )
        return {}

    # Collect pictures that need descriptions and their page numbers
    pictures: list[tuple[PictureItem, str, str | None, str]] = []
    page_numbers: set[int] = set()
    fig_counter = 0

    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        # Skip logos/icons
        classes = _get_picture_classes(item)
        if classes & _LOGO_ICON_CLASSES:
            continue

        fig_counter += 1

        caption = _get_caption_text(doc, item)
        fig_num = _extract_figure_number(caption) if caption else None
        fig_label = f"Figure {fig_num}" if fig_num else f"Figure {fig_counter}"

        if item.prov:
            page_numbers.add(item.prov[0].page_no)
        pictures.append((item, caption, fig_num, fig_label))

    if not pictures:
        return {}

    # Render only the pages that contain pictures
    logger.info(
        "Rendering %d page(s) for %d picture(s) ...",
        len(page_numbers), len(pictures),
    )
    pdf_pages = _render_pdf_pages(pdf_path, page_numbers)

    descriptions: dict[str, str] = {}

    for item, caption, fig_num, fig_label in pictures:
        image_bytes = _crop_picture_image(item, pdf_pages)
        if image_bytes is None:
            logger.debug("Skipping %s — could not crop image.", fig_label)
            continue

        # Build a prompt with figure caption context
        prompt_parts = [_VLM_SYSTEM_PROMPT]
        if caption:
            prompt_parts.append(
                f'\nThe document labels this image as: "{caption}".'
            )
        prompt = "\n".join(prompt_parts)

        logger.info("Requesting VLM description for %s ...", fig_label)
        description = _call_azure_vlm(image_bytes, prompt, config)

        if description:
            short_cap = _caption_without_prefix(caption) if caption else ""
            tag = f"[VLM - {fig_label}" + (f": {short_cap}" if short_cap else "") + "]"
            descriptions[str(item.self_ref)] = f"{tag} {description}"
            logger.info("Got VLM description for %s (%d chars).", fig_label, len(description))
        else:
            logger.warning("No VLM description returned for %s.", fig_label)

    return descriptions


# ── Pre-export element filtering ────────────────────────────────

# Labels that correspond to logos, icons, and other non-content images.
_LOGO_ICON_CLASSES: frozenset[str] = frozenset({
    "LOGO", "ICON", "STAMP", "QR_CODE", "BAR_CODE", "SIGNATURE",
})

# ── Boilerplate detection ───────────────────────────────────────

# Regex matching security-classification banner headings that appear
# on every page of Thales-template documents (e.g. "THALES GROUP INTERNAL",
# "THALES GROUP CONFIDENTIAL", "THALES GROUP RESTRICTED", etc.).
_RE_CLASSIFICATION_BANNER = re.compile(
    r"^\s*THALES\s+GROUP\s+\w+\s*$", re.IGNORECASE
)

# Regex matching Entity-ID patterns found in repeated document-ID tables
# (e.g. "15900~9950").
_RE_ENTITY_ID = re.compile(r"\d{4,6}~\d{3,5}")

# Regex matching Thales document identifiers across variants
# (e.g. "7HA-02944-AAAA-980EN", "7HA-02944-ABAA-030EN").
_RE_DOC_ID = re.compile(r"\d+[A-Z]{0,2}-\d{4,6}-[A-Z]{4}-\d{3}[A-Z]{2}")

# Headings that are empty table-of-contents / list-of-X stubs.  The PDF
# parser captures these as section headers, but their body content (the
# actual TOC entries or list items) is never extracted, leaving them as
# contentless noise.
_EMPTY_STUB_HEADINGS: frozenset[str] = frozenset({
    "contents", "table of contents",
    "list of tables", "list of figures",
    "list of abbreviations", "list of acronyms",
    "list of appendices", "list of annexes",
    "list of references", "list of documents",
    "list of drawings", "list of diagrams",
    "list of illustrations", "list of symbols",
})


# Short texts that Docling's layout model sometimes misclassifies as
# section headers.  Common in test-report templates where standalone
# result values ("Passed", "Failed") occupy their own layout region.
_FALSE_POSITIVE_HEADINGS: frozenset[str] = frozenset({
    "passed", "failed", "not applicable", "n/a", "yes", "no",
    "open", "closed", "verified", "not verified", "accepted",
    "rejected", "pending", "cancelled", "deferred", "waived",
    "completed", "incomplete", "ok", "nok", "tbd", "tbc",
})


def _is_false_positive_heading(text: str) -> bool:
    """Return True if *text* is a short non-section phrase that was
    likely misclassified as a section header by the layout model."""
    return text.strip().lower() in _FALSE_POSITIVE_HEADINGS


def _is_classification_banner(text: str) -> bool:
    """Return True if *text* looks like a security-classification heading."""
    return bool(_RE_CLASSIFICATION_BANNER.match(text.strip()))


def _is_boilerplate_table(table_item: object) -> bool:
    """Return True if a ``TableItem`` is a repeated document-ID table.

    Detection heuristic: the table contains a cell matching the
    Entity-ID pattern (``nnnnn~nnnn``) *or* the Thales document-ID
    pattern (``7HA-nnnnn-XXXX-nnnXX``).  These tables appear on every
    page as part of the page header block.
    """
    data = getattr(table_item, "data", None)
    if data is None:
        return False

    for cell in getattr(data, "table_cells", []):
        cell_text = getattr(cell, "text", "")
        if _RE_ENTITY_ID.search(cell_text) or _RE_DOC_ID.search(cell_text):
            return True
    return False


def _filter_document_elements(
    doc: DoclingDocument,
    config: ParserConfig,
    vlm_descriptions: dict[str, str] | None = None,
) -> DoclingDocument:
    """Remove noise elements from a Docling document **in-place**.

    Walks the document tree and collects elements to delete based on
    their ``DocItemLabel`` and (for pictures) their classification
    annotations.  All matching elements are removed in a single
    ``delete_items`` call at the end to avoid iterator invalidation.

    For content-bearing pictures, if ``vlm_descriptions`` contains
    a description (keyed by ``item.self_ref``), the tagged VLM text
    is inserted before deleting the picture.

    Returns the same ``doc`` reference (mutated).
    """
    from docling_core.types.doc.document import (
        DocItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
    )
    from docling_core.types.doc.labels import DocItemLabel

    vlm_map = vlm_descriptions or {}
    items_to_delete: list[NodeItem] = []
    # Deferred VLM insertions: (sibling_item, text) — applied after iteration.
    vlm_inserts: list[tuple[NodeItem, str]] = []

    for item, _level in doc.iterate_items():
        if not isinstance(item, DocItem):
            continue

        # ── Headers & footers ───────────────────────────────────
        if config.strip_headers_footers and item.label in (
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        ):
            items_to_delete.append(item)
            continue

        # ── Table of contents ───────────────────────────────────
        if config.strip_toc and item.label == DocItemLabel.DOCUMENT_INDEX:
            items_to_delete.append(item)
            continue

        # ── Boilerplate: classification banners ─────────────────
        if config.strip_boilerplate_blocks and isinstance(item, SectionHeaderItem):
            if _is_classification_banner(item.text):
                items_to_delete.append(item)
                continue

        # ── Empty stub sections (TOC, List of Figures, etc.) ───
        if config.strip_boilerplate_blocks and isinstance(item, SectionHeaderItem):
            if item.text.strip().lower() in _EMPTY_STUB_HEADINGS:
                items_to_delete.append(item)
                continue

        # ── Boilerplate: repeated document-ID tables ────────────
        if config.strip_boilerplate_blocks and isinstance(item, TableItem):
            if _is_boilerplate_table(item):
                items_to_delete.append(item)
                continue

        # ── Pictures (logos, icons, figures) ────────────────────
        if isinstance(item, PictureItem):
            classification_labels = _get_picture_classes(item)
            is_logo_or_icon = bool(classification_labels & _LOGO_ICON_CLASSES)

            if config.strip_logos_icons and is_logo_or_icon:
                items_to_delete.append(item)
                continue

            # Content-bearing picture — queue VLM description for
            # in-place insertion (done after iteration to avoid
            # mutating the tree while iterating).
            ref_key = str(item.self_ref)
            vlm_text = vlm_map.get(ref_key)

            if vlm_text:
                vlm_inserts.append((item, vlm_text))
                items_to_delete.append(item)
                continue

            # No VLM description — insert a placeholder so the
            # chunker knows a figure existed at this position.
            vlm_inserts.append(
                (item, "[Figure — see source document]")
            )
            items_to_delete.append(item)
            continue

    # ── Insert VLM descriptions in-place (before each picture) ──
    for sibling, text in vlm_inserts:
        doc.insert_text(
            sibling=sibling,
            label=DocItemLabel.PARAGRAPH,
            text=text,
            after=False,
        )

    # ── Bulk delete ─────────────────────────────────────────────
    if items_to_delete:
        doc.delete_items(node_items=items_to_delete)
        logger.info(
            "Filtered %d noise element(s) from document.",
            len(items_to_delete),
        )

    return doc


def _get_picture_classes(item: PictureItem) -> set[str]:
    """Extract classification labels from a ``PictureItem``."""
    from docling_core.types.doc.document import PictureClassificationData

    classes: set[str] = set()
    for ann in item.annotations:
        if isinstance(ann, PictureClassificationData):
            for pred in ann.predicted_classes:
                classes.add(pred.class_name.upper())
    return classes





# ── Heading hierarchy inference ─────────────────────────────────

# Regex matching a leading section number like "1", "1.1", "4.2.3", etc.
_RE_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\b")


def _infer_heading_levels(doc: DoclingDocument) -> DoclingDocument:
    """Assign heading depth based on section-numbering patterns.

    Docling's standard PDF pipeline assigns ``level=1`` to every
    ``SectionHeaderItem``, producing flat ``##`` headings.  This
    function infers hierarchy from numbering conventions:

    * ``1``       → level 1 (``##``)
    * ``1.1``     → level 2 (``###``)
    * ``1.5.1``   → level 3 (``####``)
    * ``4.2.3.1`` → level 4 (``#####``)

    Non-numbered headings (e.g. "CONTENTS", "END OF DOCUMENT") keep
    their current level (1).

    Mutates *doc* in-place and returns it.
    """
    from docling_core.types.doc.document import SectionHeaderItem

    adjusted = 0
    for item, _level in doc.iterate_items():
        if not isinstance(item, SectionHeaderItem):
            continue

        m = _RE_SECTION_NUMBER.match(item.text.strip())
        if not m:
            continue

        # Count dot-separated parts: "1" → 1, "1.1" → 2, "4.2.3" → 3
        parts = m.group(1).split(".")
        new_level = len(parts)  # 1-based: 1 → ##, 2 → ###, etc.

        if new_level != item.level:
            item.level = new_level
            adjusted += 1

    if adjusted:
        logger.info("Inferred heading levels for %d section header(s).", adjusted)

    return doc


# ── Test case restructuring ──────────────────────────────────────

# Field labels that appear in test case blocks (longest first to avoid
# partial matches during regex alternation).
_TC_FIELD_LABELS: tuple[str, ...] = (
    "Test carried out by",
    "Verification object",
    "Test verified by",
    "Failure criteria",
    "Test equipment",
    "Description",
    "Test case",
    "Test Item",
    "Reference",
    "Comment",
    "Result",
    "Date",
    "Test",
)

# Matches a test-case field label on its own line (with optional ``##``
# prefix and optional trailing colon).
_RE_TC_FIELD = re.compile(
    r"^(?:##\s+)?("
    + "|".join(re.escape(f) for f in _TC_FIELD_LABELS)
    + r")[:\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Start-of-block marker: "Test case:" on its own line.
_RE_TC_START = re.compile(r"^Test case:\s*$", re.MULTILINE)

# Numbered section heading (e.g. ``## 5 VERIFICATION``) — used to
# detect block boundaries so that section headings between test case
# blocks are not absorbed.
_RE_SECTION_BREAK = re.compile(r"^#{1,3}\s+\d+(?:\.\d+)*\s+", re.MULTILINE)

# Bracket-enclosed traceability / requirement IDs at the end of a block.
# Each bracketed token must end with digits to distinguish real IDs
# (e.g. [FVTSR_PAM_0002], [HW-IRS_PAM_266]) from document references
# (e.g. [CD_PAM], [HWADD_TOP], [DERATING]).
_RE_BRACKET_IDS = re.compile(r"^(\[[\w_./-]*\d+\])+\s*$")


def _extract_trailing_ids(value: str) -> tuple[str, list[str]]:
    """Strip bracket-enclosed IDs from the tail of a field value.

    Returns ``(cleaned_value, ids)`` where *ids* are the ``[ID]`` strings
    found at the very end of *value*.
    """
    lines = value.rstrip().split("\n")
    ids: list[str] = []

    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if _RE_BRACKET_IDS.match(last):
            ids.insert(0, last)
            lines.pop()
        else:
            break

    return "\n".join(lines).rstrip(), ids


def _try_fix_displaced_values(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Detect and fix the 'displaced value' pattern in test-case blocks.

    When Docling's layout model misaligns field labels with their values,
    the result is a run of empty fields followed by a "donor" field whose
    value contains all the displaced values as consecutive paragraphs.
    This function detects that pattern and redistributes the paragraphs
    back to the correct fields.
    """
    if len(pairs) < 3:
        return pairs

    # Count consecutive empty fields from the start.
    empty_run = 0
    for _, value in pairs:
        if value.strip():
            break
        empty_run += 1

    if empty_run < 2 or empty_run >= len(pairs):
        return pairs  # not the displaced-value pattern

    # First non-empty field is the "donor" whose value absorbed everything.
    donor_idx = empty_run
    donor_value = pairs[donor_idx][1].strip()
    if not donor_value:
        return pairs

    # Split donor value into paragraphs (blank-line separated).
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", donor_value)
        if p.strip()
    ]
    if len(paragraphs) <= 1:
        return pairs

    # Skip leading paragraphs that are not real field values:
    # orphaned bracket IDs from adjacent blocks and leaked table rows.
    skip = 0
    for p in paragraphs:
        if _RE_BRACKET_IDS.match(p) or p.lstrip().startswith("|"):
            skip += 1
        else:
            break

    usable = paragraphs[skip:]
    if len(usable) < 2:
        return pairs  # too little content to redistribute

    # Redistribute: fill as many empty fields as we have paragraphs for,
    # keeping at least one paragraph for the donor field.
    fill_count = min(empty_run, len(usable) - 1)
    new_pairs = list(pairs)
    for i in range(fill_count):
        val = usable[i]
        # Strip accidental heading markers from displaced values.
        if val.startswith("## "):
            val = val[3:].strip()
        new_pairs[i] = (pairs[i][0], val)

    # Remaining usable paragraphs stay with the donor field.
    remaining = usable[fill_count:]
    new_pairs[donor_idx] = (pairs[donor_idx][0], "\n\n".join(remaining))

    return new_pairs


def _format_single_test_block(block: str) -> str:
    """Parse a raw test-case block and return a structured markdown block.

    Returns the block wrapped in ``---`` horizontal-rule delimiters with
    each field formatted as ``**Field:** value``.  Traceability and
    requirement IDs are collected into a ``**Traceability:**`` line.
    """
    matches = list(_RE_TC_FIELD.finditer(block))
    if not matches:
        return block

    # ---- extract field → value pairs ----
    pairs: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        field_name = m.group(1).strip()
        val_start = m.end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        raw = block[val_start:val_end].strip()
        # Strip accidental heading markers on the value itself
        if raw.startswith("## "):
            raw = raw[3:].strip()
        pairs.append((field_name, raw))

    # ---- try to fix displaced values ----
    pairs = _try_fix_displaced_values(pairs)

    # ---- reject empty / malformed blocks ----
    tc_value = ""
    for name, value in pairs:
        if name.lower() == "test case":
            tc_value = value
            break
    if not tc_value:
        return block  # empty template block — pass through unchanged

    # ---- collect trailing IDs from the last field ----
    all_ids: list[str] = []
    cleaned_pairs: list[tuple[str, str]] = []
    for idx, (name, value) in enumerate(pairs):
        if idx == len(pairs) - 1:
            value, ids = _extract_trailing_ids(value)
            all_ids.extend(ids)
        cleaned_pairs.append((name, value))

    # Also pull IDs from any earlier field whose entire value is just IDs
    final_pairs: list[tuple[str, str]] = []
    for name, value in cleaned_pairs:
        stripped = value.strip()
        if stripped and all(
            _RE_BRACKET_IDS.match(ln.strip())
            for ln in stripped.split("\n")
            if ln.strip()
        ):
            # Value is purely bracket IDs — absorb into traceability
            for ln in stripped.split("\n"):
                if ln.strip():
                    all_ids.append(ln.strip())
            final_pairs.append((name, ""))
        else:
            final_pairs.append((name, value))

    # ---- build structured block ----
    lines: list[str] = ["---", ""]

    for name, value in final_pairs:
        if not value:
            continue
        # Multi-line values: put first text line on the **Field:** line
        if "\n" in value:
            first_nl = value.index("\n")
            first_line = value[:first_nl].strip()
            rest = value[first_nl + 1:]
            if first_line and not first_line.startswith("|") and not first_line.startswith("[VLM"):
                lines.append(f"**{name}:** {first_line}")
                if rest.strip():
                    lines.append(rest.rstrip())
            else:
                lines.append(f"**{name}:**")
                lines.append(value.rstrip())
        else:
            lines.append(f"**{name}:** {value}")

    if all_ids:
        lines.append(f"**Traceability:** {' '.join(all_ids)}")

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _merge_split_tables(md: str) -> str:
    """Rejoin pipe-tables that Docling split across page boundaries.

    When a table spans a page break, Docling often emits two separate
    pipe-table blocks — sometimes with the header row re-emitted in the
    second fragment.  This function detects adjacent table blocks
    separated only by blank lines, verifies they have the same column
    count, strips any duplicated header / separator from the second
    fragment, and merges them into one continuous table.
    """
    _SEP_CELL = re.compile(r"^-+$")  # e.g. "---" inside a pipe cell

    def _is_table_line(line: str) -> bool:
        return line.startswith("|")

    def _is_separator_line(line: str) -> bool:
        """True if the row is a pipe-table separator (| --- | --- | …)."""
        if not line.startswith("|"):
            return False
        cells = [c.strip() for c in line.split("|")[1:-1]]
        return bool(cells) and all(_SEP_CELL.match(c) for c in cells if c)

    def _col_count(line: str) -> int:
        return line.count("|") - 1  # leading + trailing pipes

    lines = md.split("\n")
    # Build list of (start, end) index ranges for each table block.
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]):
            start = i
            while i < len(lines) and _is_table_line(lines[i]):
                i += 1
            blocks.append((start, i))  # end is exclusive
        else:
            i += 1

    if len(blocks) < 2:
        return md

    # Walk pairs in reverse so index mutations don't invalidate later
    # pairs.
    for idx in range(len(blocks) - 1, 0, -1):
        prev_start, prev_end = blocks[idx - 1]
        cur_start, cur_end = blocks[idx]

        # Only merge if the gap between the two blocks is blank lines.
        gap = lines[prev_end:cur_start]
        if any(g.strip() for g in gap):
            continue

        # Column count must match (use first data row of each block).
        prev_cols = _col_count(lines[prev_start])
        cur_cols = _col_count(lines[cur_start])
        if prev_cols != cur_cols:
            continue

        # Determine how many leading lines of the second block to strip
        # (duplicated header row and/or separator line).
        strip = 0
        if _is_separator_line(lines[cur_start]):
            # Second block starts with separator only — strip it.
            strip = 1
        elif (
            cur_start + 1 < cur_end
            and _is_separator_line(lines[cur_start + 1])
        ):
            # Second block starts with header + separator — strip both.
            strip = 2

        # Merge: replace gap + stripped header with nothing; keep data rows.
        merged = lines[prev_end - 1 : prev_end]  # keep last row of table 1
        merged += lines[cur_start + strip : cur_end]  # data rows of table 2
        lines[prev_end - 1 : cur_end] = merged

    return "\n".join(lines)


def _restructure_test_cases(md: str) -> str:
    """Convert fragmented test-case blocks into chunking-friendly format.

    Each test case becomes a self-contained block delimited by ``---``
    horizontal rules, with ``**Field:** value`` pairs and a consolidated
    ``**Traceability:**`` metadata line.
    """
    starts = [m.start() for m in _RE_TC_START.finditer(md)]
    if not starts:
        return md

    parts: list[str] = []
    prev_end = 0

    for i, start in enumerate(starts):
        # Preserve text before this block (section headings, prose, etc.)
        parts.append(md[prev_end:start])

        next_start = starts[i + 1] if i + 1 < len(starts) else len(md)
        raw_block = md[start:next_start]

        # Check for a numbered section heading inside the block —
        # if found, the block ends there and the heading stays outside.
        skip = raw_block.find("\n") + 1
        sec_match = _RE_SECTION_BREAK.search(raw_block, pos=skip) if skip else None
        if sec_match:
            block = raw_block[: sec_match.start()]
            prev_end = start + sec_match.start()
        else:
            block = raw_block
            prev_end = next_start

        parts.append(_format_single_test_block(block))

    # Remaining text after the last test case block.
    parts.append(md[prev_end:])
    return "".join(parts)


# ── Post-export Markdown cleanup ─────────────────────────────────

# Regex patterns compiled once at module level.
_RE_EXCESSIVE_BLANKS = re.compile(r"\n{3,}")
_RE_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_RE_PAGE_X_OF_Y = re.compile(
    r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE
)

# Orphaned timestamp lines extracted by OCR from screenshot metadata
# embedded in PDF pages (e.g. "2014", "0845", "Sep 2014 11:28:53",
# "4 Sep 2014 14:23:08").  These sit on their own line and carry no
# semantic value.
_RE_ORPHANED_TIMESTAMP = re.compile(
    r"^\s*"
    r"(?:"
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"  # "4 Sep 2014 14:23:08"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"          # "Sep 2014 11:28:53"
    r"|\d{4}"                                                                                                       # bare year "2014" or time "0845"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_RE_REPEATED_SEPARATORS = re.compile(
    r"(^[ \t]*[-=_]{3,}[ \t]*$\s*){2,}", re.MULTILINE
)

# Build a regex that matches headings whose text is a known false-positive
# (e.g. "## Passed", "## Not applicable").  These are test-report result
# values that Docling's layout model misclassified as section headers.
# The regex demotes them to plain text (keeps the value, strips the "## ").
_RE_FALSE_POSITIVE_HEADING = re.compile(
    r"^##\s+(?:" + "|".join(re.escape(h) for h in sorted(_FALSE_POSITIVE_HEADINGS)) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# PDF line-wrap hyphenation artifact: a lowercase letter followed by a hyphen,
# a single space, then a lowercase continuation ("daugh- terboard" → "daughterboard").
# Only fires when BOTH sides are lowercase to avoid breaking identifiers like
# "HW- IRS_DIM_448" or compound proper nouns like "Documentation- Plan".
_RE_HYPHENATION_ARTIFACT = re.compile(r"([a-z])- ([a-z])")

# Table captions promoted to headings by Docling's layout model:
# "## Table 2: Additional references" → "Table 2: Additional references".
_RE_TABLE_HEADING = re.compile(
    r"^#{1,6}\s+(Table\s+\d+\s*:.*)$",
    re.MULTILINE,
)

# Thales copyright / reproduction notice that appears on cover pages and
# occasionally mid-document as a running footer.
_RE_THALES_COPYRIGHT = re.compile(
    r"^This document is not to be reproduced.*?\u00a9\s*THALES\s+\d{4}\s*-\s*All\s+rights\s+reserved\.\s*$",
    re.MULTILINE,
)

# Terminal / export boundary markers: "## END OF DOCUMENT", bare
# "END OF DOCUMENT", "(Start Of Doors Export)", "(End Of Doors Export)".
_RE_TERMINAL_MARKER = re.compile(
    r"^(?:#{1,6}\s+)?(?:END\s+OF\s+DOCUMENT|\((?:Start|End)\s+Of\s+Doors\s+Export\))\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Thales-template page header that appears on every page of DOORS-exported
# documents: "## PROJECT ACRONYM DOCUMENT TITLE".
_RE_PROJECT_ACRONYM_HEADING = re.compile(
    r"^#{1,6}\s+PROJECT\s+ACRONYM\s+DOCUMENT\s+TITLE\s*$",
    re.MULTILINE,
)

# Safety-net regex: catches any residual boilerplate blocks that survived
# the pre-export filter.  Matches a classification banner heading followed
# by a document-ID table (with pipe-delimited rows and separator line).
_RE_RESIDUAL_BOILERPLATE = re.compile(
    r"^#{1,6}\s+THALES\s+GROUP\s+\w+\s*\n"   # "## THALES GROUP INTERNAL"
    r"(?:\s*\n)*"                               # optional blank lines
    r"\|.+\|\s*\n"                              # table header row
    r"\|[-| ]+\|\s*\n"                          # table separator row
    r"(?:\|.+\|\s*\n)*",                        # remaining table rows
    re.IGNORECASE | re.MULTILINE,
)

# Front-matter address block at the start of Thales-template documents.
# Matches a company name heading followed by address, phone, fax, and URL lines.
_RE_FRONT_MATTER = re.compile(
    r"\A\s*"                                    # start of document
    r"(?:#{1,6}\s+.+\n)?"                       # optional heading (company name)
    r"(?:.*\n){0,5}?"                            # up to 5 lines of address
    r"(?=.*(?:Tel\.|Fax\.|www\.))"               # lookahead: block must contain phone/fax/URL
    r"(?:.*\n)*?"                                # consume lines
    r"(?:.*www\.\S+.*\n)",                       # final line with URL
    re.IGNORECASE,
)


# ── DOORS metadata condensation ──────────────────────────────────

# DOORS export field labels in their expected order of appearance.
_DOORS_FIELDS: tuple[str, ...] = (
    "Category", "Allocation", "Priority", "Safety", "Verification",
    "Comment", "No_Out-Links",
)

_DOORS_CORE_FIELDS: tuple[str, ...] = (
    "Category", "Allocation", "Priority", "Safety", "Verification",
)

# Matches a DOORS field label at the start of a line.
_RE_DOORS_FIELD = re.compile(
    r"^(" + "|".join(re.escape(f) for f in _DOORS_FIELDS) + r"):\s*(.*?)\s*$"
)

# Template page header that can intrude mid-block.
_RE_PROJECT_ACRONYM = re.compile(
    r"^#{1,6}\s+PROJECT\s+ACRONYM\s+DOCUMENT\s+TITLE\s*$"
)

# Known DOORS verification method values.
_DOORS_VERIFICATION_METHODS: frozenset[str] = frozenset({
    "Analysis", "Test", "Inspection", "Demonstration",
})


def _condense_doors_metadata(md: str) -> str:
    """Condense verbose DOORS-exported metadata blocks into inline tags.

    DOORS exports produce a repeating block of metadata fields after every
    requirement (Category, Allocation, Priority, Safety, Verification),
    each on its own line separated by blank lines.  This function detects
    those blocks and replaces each with a single bracketed inline tag::

        [Category: Requirement | Allocation: HW | Priority: Mandatory
         | Safety: Yes | Verification: Analysis]

    Traceability IDs (e.g. ``[HWADD:TOP:0012]``) are preserved on a
    separate line immediately after the tag.
    """
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    condensed = 0

    while i < n:
        # ── Detect block start: bare "Category:" line ───────────
        cat_m = re.match(r"^Category:\s*(.*?)\s*$", lines[i])
        if not cat_m:
            out.append(lines[i])
            i += 1
            continue

        block_start = i
        fields: dict[str, str] = {}
        trace_ids: list[str] = []
        cur_field: str | None = "Category"
        cur_val: str = cat_m.group(1)
        i += 1

        while i < n:
            stripped = lines[i].strip()

            # Skip blank lines within the block
            if not stripped:
                i += 1
                continue

            # Skip template page-header noise within the block
            if _RE_PROJECT_ACRONYM.match(stripped):
                i += 1
                continue

            # ── New DOORS field label ───────────────────────────
            fm = _RE_DOORS_FIELD.match(stripped)
            if fm:
                if cur_field:
                    fields[cur_field] = cur_val.strip()
                new_name = fm.group(1)
                rest = fm.group(2)

                # Handle collapsed fields on one line, e.g.
                # "Safety: Yes Verification: Analysis [ID]"
                scanning = True
                while scanning:
                    scanning = False
                    for cf in _DOORS_FIELDS:
                        inner = re.search(
                            r"\s+" + re.escape(cf) + r":\s*", rest,
                        )
                        if inner:
                            fields[new_name] = rest[: inner.start()].strip()
                            new_name = cf
                            rest = rest[inner.end():]
                            scanning = True
                            break

                # Pull inline bracket IDs from the remainder
                for tid in re.findall(r"\[[\w_./:|-]+\]", rest):
                    rest = rest.replace(tid, "", 1)
                    trace_ids.append(tid)

                cur_field = new_name
                cur_val = rest.strip()
                i += 1
                continue

            # ── Traceability ID line ────────────────────────────
            if stripped.startswith("[") and re.match(
                r"^\[[\w_./:|-]+\]", stripped,
            ):
                if cur_field:
                    fields[cur_field] = cur_val.strip()
                    cur_field = None
                    cur_val = ""
                for tid in re.findall(r"\[[\w_./:|-]+\]", stripped):
                    trace_ids.append(tid)
                i += 1
                continue

            # ── Value for the current (empty-value) field ───────
            if cur_field and not cur_val:
                cur_val = stripped
                i += 1
                continue

            # ── Extra verification methods (Analysis, Test, …) ──
            if cur_field == "Verification" and cur_val:
                if stripped in _DOORS_VERIFICATION_METHODS:
                    cur_val += ", " + stripped
                    i += 1
                    continue
                if stripped == "NA":
                    i += 1   # trailing noise (Coverage_ID / No_Out-Links)
                    continue

            # ── Trailing bare "NA" after all fields collected ───
            if cur_field is None and stripped == "NA":
                i += 1
                continue

            # ── Block boundary — stop collecting ────────────────
            break

        # Save last field
        if cur_field:
            fields[cur_field] = cur_val.strip()

        # Validate: need >= 3 core fields for a real DOORS block
        core_count = sum(1 for f in _DOORS_CORE_FIELDS if fields.get(f))
        if core_count >= 3:
            parts = [
                f"{f}: {fields[f]}"
                for f in _DOORS_CORE_FIELDS
                if fields.get(f)
            ]
            for f in ("Comment", "No_Out-Links"):
                v = fields.get(f, "").strip()
                if v and v.upper() != "NA":
                    parts.append(f"{f}: {v}")
            out.append("[" + " | ".join(parts) + "]")
            if trace_ids:
                out.append(" ".join(trace_ids))
            out.append("")  # blank line separator after the condensed tag
            condensed += 1
        else:
            # Not a valid DOORS block — emit original lines unchanged
            out.extend(lines[block_start:i])

    if condensed:
        logger.info("Condensed %d DOORS metadata block(s).", condensed)

    return "\n".join(out)


# Regex matching DOORS requirement / object IDs on their own line.
# Examples: "HW-IRS_DIM_VI_275", "TOP_SRS_1500".
_RE_DOORS_REQ_ID = re.compile(r"^(?:#{1,6}\s+)?([A-Z][A-Z0-9]*[-_][\w-]*_\d+)\s*$")

# Matches lines consisting entirely of bracketed traceability IDs.
_RE_TRACE_IDS_LINE = re.compile(r"^(?:\[[\w_./:|-]+\]\s*)+$")


def _restructure_doors_requirements(md: str) -> str:
    """Wrap DOORS requirement blocks in ``---`` delimiters for atomic chunking.

    After ``_condense_doors_metadata()`` has collapsed verbose metadata
    into ``[Category: ...]`` inline tags, this function identifies each
    complete requirement block (ID + text + metadata + traceability IDs)
    and wraps it in horizontal-rule delimiters so the chunker treats each
    requirement as a self-contained unit.
    """
    # Quick exit: only process documents that contain condensed DOORS tags.
    if "[Category:" not in md:
        return md

    lines = md.split("\n")
    n = len(lines)

    # ── Pass 1: locate all [Category: ...] tag lines ────────────────
    tag_indices: list[int] = []
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[Category:") and s.endswith("]"):
            tag_indices.append(idx)

    if not tag_indices:
        return md

    # ── Pass 2: determine block boundaries for each tag ─────────
    blocks: list[tuple[int, int]] = []  # (start, end) inclusive

    for tag_idx in tag_indices:
        # ── Forward: collect trailing traceability ID lines ─────
        end = tag_idx
        j = tag_idx + 1
        while j < n:
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            if (
                s.startswith("[")
                and not s.startswith("[Category:")
                and _RE_TRACE_IDS_LINE.match(s)
            ):
                end = j
                j += 1
                continue
            break

        # ── Backward: find block start (text, then req IDs) ──────
        start = tag_idx
        j = tag_idx - 1
        phase = "text"  # first collect text lines, then req IDs

        while j >= 0:
            s = lines[j].strip()

            if not s:
                j -= 1
                continue

            # Hard boundaries: never cross these
            if (
                s.startswith("|")           # table row
                or s == "---"               # previous block delimiter
                or (s.startswith("[Category:") and s.endswith("]"))
                or _RE_TRACE_IDS_LINE.match(s)
            ):
                break

            # Section heading: don't absorb into the block
            if re.match(r"^#{1,6}\s+", s) and not _RE_DOORS_REQ_ID.match(s):
                break

            if phase == "text":
                if _RE_DOORS_REQ_ID.match(s):
                    phase = "req_ids"
                    start = j
                    j -= 1
                    continue
                # Regular text line — include in block
                start = j
                j -= 1
                continue

            # phase == "req_ids": collect consecutive DOORS IDs
            if _RE_DOORS_REQ_ID.match(s):
                start = j
                j -= 1
                continue
            # Non-ID line after IDs → block boundary
            break

        # Safety: don't overlap with previous block
        if blocks:
            start = max(start, blocks[-1][1] + 1)

        blocks.append((start, end))

    # ── Pass 3: rebuild Markdown with structured blocks ──────────
    out: list[str] = []
    prev_end = -1

    for start, end in blocks:
        # Emit unprocessed lines before this block
        for k in range(prev_end + 1, start):
            out.append(lines[k])

        # Find [Category:] tag within the block
        tag_offset: int | None = None
        for k in range(start, end + 1):
            s = lines[k].strip()
            if s.startswith("[Category:") and s.endswith("]"):
                tag_offset = k
                break

        if tag_offset is None:
            out.extend(lines[start : end + 1])
            prev_end = end
            continue

        # ── Extract components ─────────────────────────────────
        req_ids: list[str] = []
        text_lines: list[str] = []

        for k in range(start, tag_offset):
            s = lines[k].strip()
            if not s:
                if text_lines:
                    text_lines.append(lines[k])
                continue
            m = _RE_DOORS_REQ_ID.match(s)
            if m and not text_lines:
                req_ids.append(m.group(1))  # strip any leading ## prefix
            else:
                text_lines.append(lines[k])

        # Trim trailing blank lines from text
        while text_lines and not text_lines[-1].strip():
            text_lines.pop()

        tag_line = lines[tag_offset].strip()

        # Traceability IDs after the tag
        trace_parts: list[str] = []
        for k in range(tag_offset + 1, end + 1):
            s = lines[k].strip()
            if s:
                trace_parts.append(s)

        # ── Emit structured block ───────────────────────────────
        out.append("")
        out.append("---")
        out.append("")
        for rid in req_ids:
            out.append(f"**{rid}**")
        if req_ids:
            out.append("")
        for tl in text_lines:
            out.append(tl)
        if text_lines:
            out.append("")
        out.append(tag_line)
        if trace_parts:
            out.append("**Traceability:** " + " ".join(trace_parts))
        out.append("")
        out.append("---")
        out.append("")

        prev_end = end

    # Emit remaining lines after the last block
    for k in range(prev_end + 1, n):
        out.append(lines[k])

    logger.info(
        "Structured %d DOORS requirement block(s) with delimiters.",
        len(blocks),
    )

    return "\n".join(out)


def _clean_markdown(md: str) -> str:
    """Post-process a raw Markdown string for chunking quality.

    Applies the following transforms in order:

    1. Strip front-matter address block (company name, phone, fax, URL).
    2. Remove residual "Page X of Y" lines.
    3. Remove residual boilerplate blocks (classification banner + doc-ID table).
    3b. Strip ``## PROJECT ACRONYM DOCUMENT TITLE`` page-header noise.
    3c. Rejoin words broken by PDF line-wrap hyphenation
        (``daugh- terboard`` → ``daughterboard``).
    3d. Strip Thales copyright / reproduction notice.
    3e. Remove terminal / export boundary markers
        (``END OF DOCUMENT``, ``(Start/End Of Doors Export)``).
    3f. Merge pipe-tables split across page boundaries.
    4. Demote false-positive headings ("## Passed" → "Passed").
    4b. Demote ``## Table N: ...`` captions to plain text.
    5. Condense DOORS-exported metadata blocks into inline tags.
    5b. Wrap DOORS requirement blocks in ``---`` delimiters for
        atomic chunking.
    6. Restructure test-case blocks into ``**Field:** value`` format with
       ``---`` delimiters and consolidated traceability metadata.
    7. Collapse repeated separator lines (``---``, ``===``, ``___``).
    8. Decode residual HTML entities (``&amp;`` → ``&``).
    9. Strip HTML comment artifacts (``<!-- ... -->``).
    10. Strip trailing whitespace from every line.
    11. Collapse 3+ consecutive blank lines down to 2.
    12. Strip leading / trailing whitespace from the whole document.
    """
    # 1. Strip front-matter address block
    md = _RE_FRONT_MATTER.sub("", md)

    # 2. Remove "Page X of Y" artifacts
    md = _RE_PAGE_X_OF_Y.sub("", md)

    # 2b. Remove orphaned timestamp lines (OCR screenshot metadata)
    md = _RE_ORPHANED_TIMESTAMP.sub("", md)

    # 3. Remove residual boilerplate blocks
    md = _RE_RESIDUAL_BOILERPLATE.sub("", md)

    # 3b. Strip "## PROJECT ACRONYM DOCUMENT TITLE" page-header noise
    md = _RE_PROJECT_ACRONYM_HEADING.sub("", md)

    # 3c. Rejoin words broken by PDF line-wrap hyphenation
    md = _RE_HYPHENATION_ARTIFACT.sub(r"\1\2", md)

    # 3d. Strip Thales copyright / reproduction notice
    md = _RE_THALES_COPYRIGHT.sub("", md)

    # 3e. Remove terminal / export boundary markers
    md = _RE_TERMINAL_MARKER.sub("", md)

    # 3f. Merge pipe-tables split across page boundaries
    md = _merge_split_tables(md)

    # 4. Demote false-positive headings to plain text
    md = _RE_FALSE_POSITIVE_HEADING.sub(
        lambda m: m.group(0).lstrip("# ").strip(), md
    )

    # 4b. Demote "## Table N: ..." captions to plain text
    md = _RE_TABLE_HEADING.sub(r"\1", md)

    # 5. Condense DOORS-exported metadata blocks into inline tags
    md = _condense_doors_metadata(md)

    # 5b. Wrap DOORS requirement blocks in --- delimiters
    md = _restructure_doors_requirements(md)

    # 6. Restructure test-case blocks for chunking
    md = _restructure_test_cases(md)

    # 7. Collapse repeated separator lines into a single one
    md = _RE_REPEATED_SEPARATORS.sub("---\n", md)

    # 8. Decode residual HTML entities
    md = html_module.unescape(md)

    # 9. Strip HTML comment artifacts
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)

    # 10. Strip trailing whitespace per line
    md = _RE_TRAILING_WHITESPACE.sub("", md)

    # 11. Collapse excessive blank lines
    md = _RE_EXCESSIVE_BLANKS.sub("\n\n", md)

    # 12. Strip leading/trailing whitespace from the whole document
    md = md.strip()

    return md


# ── Main entry point ────────────────────────────────────────────


def parse_document(
    path: Path,
    doc_id: str,
    config: ParserConfig | None = None,
    converter: DocumentConverter | None = None,
) -> ParsedDocument | None:
    """Parse a single document to clean Markdown.

    Parameters
    ----------
    path:
        Path to the source PDF or DOCX file.
    doc_id:
        SHA-256 hex digest (from the ingestion registry).
    config:
        Parser configuration.  Uses ``ParserConfig()`` defaults if omitted.
    converter:
        Pre-built ``DocumentConverter``.  If omitted, one is built from
        *config*.  Pass a shared converter for batch runs to avoid
        reloading models for every file.

    Returns
    -------
    ParsedDocument
        On success — contains the cleaned Markdown and metadata.
    None
        On failure — the error is logged, caller decides how to handle.
    """
    from docling.datamodel.base_models import ConversionStatus

    from ..config.paths import MARKDOWN_DIR

    cfg = config or ParserConfig()

    # ── 1. Build or reuse converter ─────────────────────────────
    conv = converter or _build_converter(cfg)

    # ── 2. Convert document via Docling ─────────────────────────
    try:
        result = conv.convert(source=path, raises_on_error=False)
    except Exception:
        logger.exception("Docling conversion failed for '%s'.", path.name)
        return None

    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        # Docling may report FAILURE even when most pages were extracted.
        # Proceed if the document object contains any usable content.
        doc = result.document
        has_content = doc is not None and any(True for _ in doc.iterate_items())
        if has_content:
            logger.warning(
                "Conversion returned status '%s' for '%s' — "
                "proceeding with partially extracted content.",
                result.status.name,
                path.name,
            )
        else:
            logger.error(
                "Conversion returned status '%s' for '%s' with no usable content.",
                result.status.name,
                path.name,
            )
            return None
    else:
        doc = result.document

    # ── 3. Extract metadata before filtering ────────────────────
    page_count = result.input.page_count

    # Title: prefer the document's own name, fall back to first heading
    title = doc.name or ""
    if not title:
        from docling_core.types.doc.labels import DocItemLabel

        for item, _ in doc.iterate_items():
            label = getattr(item, "label", None)
            if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
                title = getattr(item, "text", "") or ""
                if title:
                    break

    if not title:
        title = path.stem

    # ── 3b. VLM figure descriptions (Azure GPT-4.1) ────────────
    vlm_descriptions = _describe_pictures(doc, path, cfg)

    # ── 4. Filter noise elements ────────────────────────────────
    _filter_document_elements(doc, cfg, vlm_descriptions=vlm_descriptions)

    # ── 4b. Infer heading hierarchy from section numbering ──────
    _infer_heading_levels(doc)

    # ── 5. Export to Markdown ───────────────────────────────────
    raw_md = doc.export_to_markdown(
        escape_underscores=False,
        escape_html=False,
    )

    # ── 6. Clean the Markdown ───────────────────────────────────
    cleaned_md = _clean_markdown(raw_md)

    if not cleaned_md:
        logger.warning("Parsing produced empty Markdown for '%s'.", path.name)
        return None

    # ── 7. Write to cache ───────────────────────────────────────
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MARKDOWN_DIR / f"{doc_id}.md"
    out_path.write_text(cleaned_md, encoding="utf-8")
    logger.info(
        "Saved parsed Markdown for '%s' → '%s' (%d chars).",
        path.name,
        out_path.name,
        len(cleaned_md),
    )

    # ── 8. Return result ────────────────────────────────────────
    return ParsedDocument(
        doc_id=doc_id,
        source_path=path,
        markdown=cleaned_md,
        title=title,
        page_count=page_count,
        metadata={
            "source_format": result.input.format.name if result.input.format else "UNKNOWN",
            "conversion_status": result.status.name,
            "file_size_bytes": result.input.filesize,
        },
    )
