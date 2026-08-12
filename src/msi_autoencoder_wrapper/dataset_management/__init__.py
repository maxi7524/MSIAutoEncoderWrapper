"""Compatibility exports for the independent :mod:`msi_dataset_manager`."""

from msi_dataset_manager.catalog import DatasetCatalog
from msi_dataset_manager.operations import import_local_dataset

__all__ = ["DatasetCatalog", "import_local_dataset"]
