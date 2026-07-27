"""External MSI database query and materialization interfaces."""

from .base_source import DatasetSource
from .source_manager import DatasetSourceManager

__all__ = ["DatasetSource", "DatasetSourceManager"]
