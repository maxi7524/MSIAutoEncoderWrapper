"""SQLite schema for one composed MSI annotation store."""

from __future__ import annotations

import sqlite3


REFERENCE_TABLE_PREFIX = "reference_annotations_"


def create_merged_annotation_schema(connection: sqlite3.Connection) -> None:
    """Create the fixed global tables of a merged annotation store."""
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE datasets_metadata (
            dataset_index INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            source_dataset_id TEXT NOT NULL,
            name TEXT,
            source_imzml_path TEXT NOT NULL,
            reference_table_name TEXT NOT NULL UNIQUE,
            filtering_metadata_json TEXT NOT NULL,
            dataset_metadata_json TEXT NOT NULL,
            UNIQUE(source, source_dataset_id)
        );

        CREATE TABLE pixel_segments (
            segment_id INTEGER PRIMARY KEY,
            dataset_index INTEGER NOT NULL,
            merged_pixel_start INTEGER NOT NULL,
            segment_length INTEGER NOT NULL CHECK(segment_length > 0),
            source_pixel_start INTEGER NOT NULL,
            source_step INTEGER NOT NULL,
            FOREIGN KEY(dataset_index)
                REFERENCES datasets_metadata(dataset_index)
        );

        CREATE TABLE merged_annotations (
            merged_annotation_id INTEGER PRIMARY KEY,
            formula TEXT NOT NULL,
            adduct TEXT NOT NULL,
            charge INTEGER,
            pixel_indices_blob BLOB NOT NULL,
            UNIQUE(formula, adduct)
        );

        CREATE INDEX pixel_segments_merged_lookup
            ON pixel_segments(merged_pixel_start);
        CREATE INDEX merged_annotations_identity
            ON merged_annotations(formula, adduct);
        """
    )


def create_reference_annotation_table(
    connection: sqlite3.Connection,
    dataset_index: int,
) -> str:
    """Create and return the generated reference table for one source dataset."""
    table_name = reference_table_name(dataset_index)
    connection.executescript(
        f"""
        CREATE TABLE {table_name} (
            reference_annotation_id INTEGER PRIMARY KEY,
            merged_annotation_id INTEGER NOT NULL,
            source_annotation_id TEXT NOT NULL,
            formula TEXT NOT NULL,
            adduct TEXT NOT NULL,
            mz REAL,
            fdr REAL,
            database_id TEXT,
            database_name TEXT,
            database_version TEXT,
            source_record_json TEXT NOT NULL,
            UNIQUE(source_annotation_id),
            FOREIGN KEY(merged_annotation_id)
                REFERENCES merged_annotations(merged_annotation_id)
        );

        CREATE INDEX {table_name}_merged_lookup
            ON {table_name}(merged_annotation_id);
        CREATE INDEX {table_name}_fdr_lookup
            ON {table_name}(fdr);
        """
    )
    return table_name


def reference_table_name(dataset_index: int) -> str:
    """Return the validated generated reference table name."""
    value = int(dataset_index)
    if value <= 0:
        raise ValueError("dataset_index must be greater than zero")
    return f"{REFERENCE_TABLE_PREFIX}{value:04d}"
