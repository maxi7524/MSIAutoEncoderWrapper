"""Synthetic sampling strategy implementations loaded through discovery."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

discover_modules(__name__, recursive=False)
