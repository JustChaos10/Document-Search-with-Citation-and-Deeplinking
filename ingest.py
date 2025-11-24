from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import AppConfig
from app.ingestion import IngestionPipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific file paths to process. Defaults to ingesting the contents of the uploads directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    pipeline = IngestionPipeline(config)

    target_files = [Path(f) for f in args.files] if args.files else None
    summary = pipeline.ingest(target_files=target_files)

    print(
        f"Ingestion complete. Files: {summary.files_processed}, "
        f"sections: {summary.sections_extracted}, chunks: {summary.chunks_written}"
    )


if __name__ == "__main__":
    main()
