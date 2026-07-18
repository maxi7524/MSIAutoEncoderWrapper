"""Global spatial coordinate convention for image-like data sources."""

from __future__ import annotations

from typing import Any, Literal

from ...utils.exceptions import raise_validation_error

CoordinateOrder = Literal["xy", "matrix"]


class SpatialContextMixin:
    """Expose a wrapper-wide coordinate order used by image and latent readers."""

    def __init__(
        self,
        coordinate_order: CoordinateOrder = "xy",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the global spatial convention.

        :param coordinate_order: ``xy`` for image coordinates or ``matrix`` for
            row-column coordinates where the first two axes are reversed.
        :type coordinate_order: Literal["xy", "matrix"]
        """
        self.coordinate_order: CoordinateOrder = "xy"
        self.set_coordinate_order(coordinate_order)
        super().__init__(*args, **kwargs)

    def set_coordinate_order(self, coordinate_order: CoordinateOrder) -> None:
        """Change the coordinate convention for all spatial user APIs.

        :param coordinate_order: ``xy`` or ``matrix``.
        :type coordinate_order: Literal["xy", "matrix"]
        :raises ValidationError: If the convention is unsupported.
        """
        if coordinate_order not in {"xy", "matrix"}:
            raise_validation_error(
                context_name="SpatialContext",
                message="coordinate_order must be either 'xy' or 'matrix'.",
            )
        self.coordinate_order = coordinate_order
