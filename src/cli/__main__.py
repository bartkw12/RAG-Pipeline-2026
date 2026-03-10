"""Allow running the ingest CLI as ``python -m src.cli.ingest``."""

from src.cli.ingest import main
import sys

sys.exit(main())
