"""CLI entrypoint for the ingestion pipeline.

Usage examples::

    # Drop-folder mode (default) — ingest everything in input/
    python -m src.cli.ingest

    # CLI mode — explicit files or globs
    python -m src.cli.ingest --paths "D:/specs/*.pdf" report.docx

    # Manifest mode
    python -m src.cli.ingest --manifest manifest.json

    # Dry run — see what would happen without doing anything
    python -m src.cli.ingest --dry-run

    # Force re-ingest everything, ignore duplicate detection
    python -m src.cli.ingest --force

    # Override the default input directory
    python -m src.cli.ingest --input-dir "D:/Engineering/Docs"

    # Combine flags
    python -m src.cli.ingest --paths "*.pdf" --dry-run --verbose
"""

