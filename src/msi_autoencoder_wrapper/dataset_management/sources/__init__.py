"""External database source contracts and strategy registry."""

from .base import DatasetSource
from .source_manager import DatasetSourceManager

__all__ = ["DatasetSource", "DatasetSourceManager"]
