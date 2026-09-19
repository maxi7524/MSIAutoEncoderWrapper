"""Registered synthetic spectrum representation strategies."""

from .base import (
    SyntheticRepresentationContext,
    SyntheticRepresentationManager,
    SyntheticRepresentationSpec,
    SyntheticRepresentationStrategy,
    get_representation_strategy,
    register_representation_strategy,
)

__all__ = [
    "SyntheticRepresentationContext",
    "SyntheticRepresentationManager",
    "SyntheticRepresentationSpec",
    "SyntheticRepresentationStrategy",
    "get_representation_strategy",
    "register_representation_strategy",
]
