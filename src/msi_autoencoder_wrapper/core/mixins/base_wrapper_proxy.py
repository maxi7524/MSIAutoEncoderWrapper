# Heading 1 (Base Wrapper Proxy Definition)
## Global base class to unify wrapper access across all functional proxies

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...core.wrapper import MSIAutoEncoderWrapper

class BaseWrapperProxy:
    """
    Common base proxy giving unified, type-safe access to the shared Wrapper state.
    """
    def __init__(self, wrapper_ref: Any) -> None:
        """
        Initializes the base proxy with a reference to the master wrapper.

        :param wrapper_ref: Loose reference back to the coordinating master wrapper facade.
        :type wrapper_ref: Any
        """
        self._wrapper: "MSIAutoEncoderWrapper" = wrapper_ref

    @property
    def project_path(self) -> str:
        """
        Returns the project root path registered on the wrapper.

        :return: Absolute project path string.
        :rtype: str
        """
        return getattr(self._wrapper, "_project_path", "")

    @property
    def active_image_key(self) -> Optional[str]:
        """
        Safely extracts the active image key from the active context layer.

        :return: The currently active image identifier, or None.
        :rtype: Optional[str]
        """
        active_ctx = getattr(self._wrapper, "active_context", None)
        if active_ctx is not None:
            return getattr(active_ctx, "_instantiated_image_key", None)
        return None

    @property
    def active_model_name_global(self) -> Optional[str]:
        """
        Safely extracts the active model name from the model manager layer.

        :return: The active model name, or None.
        :rtype: Optional[str]
        """
        models_mngr = getattr(self._wrapper, "models_manager", None)
        if models_mngr is not None:
            return getattr(models_mngr, "_active_model_name", None)
        return None