"""Architecture implementation registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

discover_modules(__name__, excluded_parts={"schema"})
