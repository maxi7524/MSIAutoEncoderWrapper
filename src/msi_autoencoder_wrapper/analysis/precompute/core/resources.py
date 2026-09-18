"""Shared in-process resources for an ordered precompute strategy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class ResourceManager:
    """Memoize resources requested by several analysis plugins.

    The manager is intentionally generic. Model-aware plugins use an immutable
    checkpoint identity as their key, while other plugins can share decoded splits,
    catalogue objects, or prepared tensors. A resource is constructed at most once in
    one strategy run unless a plugin explicitly releases it for memory reasons.
    """

    def __init__(self) -> None:
        self._resources: dict[str, Any] = {}

    def get_or_create(self, key: str, factory: Callable[[], T]) -> T:
        """Return a memoized resource, constructing it only on the first request."""
        if key not in self._resources:
            self._resources[key] = factory()
        return self._resources[key]

    def get(self, key: str) -> Any:
        """Return an already materialized resource.

        :raises KeyError: If the resource has not been registered.
        """
        return self._resources[key]

    def release(self, key: str) -> None:
        """Release one resource when an explicit memory boundary is reached."""
        self._resources.pop(key, None)

    def clear(self) -> None:
        """Release every in-process resource after the strategy completes."""
        active = self._resources.get("__active_model_key__")
        if active is not None and active in self._resources:
            self._resources[active].to("cpu")
        self._resources.clear()

    def activate_model(self, key: str, factory: Callable[[], T], device: str) -> T:
        """Load a model once, keeping only the active checkpoint on the accelerator.

        :param key: Immutable checkpoint identity.
        :type key: str
        :param factory: Loader called only when the checkpoint has not been seen.
        :type factory: collections.abc.Callable[[], T]
        :param device: Target accelerator/device accepted by ``torch.nn.Module.to``.
        :type device: str
        :return: The active model instance.
        :rtype: T

        Models remain resident on CPU after their first use. This prevents repeated
        artifact deserialization across stages without retaining every checkpoint on a
        limited GPU. Analysis plugins execute serially, so one active model is enough.
        """
        model = self.get_or_create(key, factory)
        active = self._resources.get("__active_model_key__")
        if active != key:
            if active is not None and active in self._resources:
                self._resources[active].to("cpu")
            model.to(device)
            self._resources["__active_model_key__"] = key
        model.eval()
        return model
