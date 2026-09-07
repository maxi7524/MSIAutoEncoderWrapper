"""Enrich normalized JSON annotations or an existing merged SQLite store."""

import argparse
import json
from pathlib import Path
import re
import sqlite3

from . import LipidMapsProvider, SnapshotProvider, enrich_annotations


def main() -> None:
    """Run explicit enrichment, preserving the existing annotation representation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--lipidmaps-version")
    args = parser.parse_args()
    if args.annotations.suffix == ".json":
        records = json.loads(args.annotations.read_text())
    else:
        with sqlite3.connect(args.annotations.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            records = [dict(row) for row in connection.execute("SELECT formula, adduct FROM merged_annotations")]
            for row in connection.execute("SELECT reference_table_name FROM datasets_metadata"):
                table = row[0]
                if not re.fullmatch(r"reference_annotations_[0-9]+", table):
                    raise ValueError("Invalid reference annotation table name.")
                for reference in connection.execute(f"SELECT formula, adduct, source_record_json FROM {table}"):
                    records.append({"formula": reference[0], "adduct": reference[1], "source_record": json.loads(reference[2])})
    provider = SnapshotProvider(args.snapshot) if args.snapshot else LipidMapsProvider(args.lipidmaps_version)
    enrich_annotations(records, args.output, provider)


if __name__ == "__main__":
    main()
