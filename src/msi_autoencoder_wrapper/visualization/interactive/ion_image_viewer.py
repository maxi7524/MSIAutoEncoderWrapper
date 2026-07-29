"""Interactive traversal of aligned ion-image collections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..spatial import plot_image_grid
from ..theme import VisualizationTheme, resolve_theme


class IonImageViewer:
    """Display lazily generated aligned images while traversing an m/z axis.

    :param mass_axis: Available m/z values.
    :type mass_axis: Sequence[float]
    :param image_provider: Callable returning named spatial arrays for one m/z.
    :type image_provider: Callable[[float], Mapping[str, numpy.ndarray]]
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    """

    def __init__(
        self,
        mass_axis: Sequence[float],
        image_provider: Callable[[float], Mapping[str, np.ndarray]],
        theme: VisualizationTheme | str | None = None,
    ) -> None:
        self.mass_axis = np.asarray(mass_axis, dtype=np.float64)
        self.image_provider = image_provider
        self.theme = resolve_theme(theme)
        if self.mass_axis.ndim != 1 or self.mass_axis.size == 0:
            raise ValueError("mass_axis must contain at least one m/z value.")

    def plot(self, index: int = 0) -> Any:
        """Render one indexed m/z image collection.

        :param index: Zero-based mass-axis index.
        :type index: int
        :return: Figure and axes returned by the spatial grid renderer.
        :rtype: Any
        """
        mass = float(self.mass_axis[index])
        images = self.image_provider(mass)
        figure, axes = plot_image_grid(images, theme=self.theme)
        figure.suptitle(f"m/z {mass:.5f}")
        return figure, axes

    def widget(self, initial_index: int = 0) -> Any:
        """Return an ipywidgets slider that updates one reusable output area.

        :param initial_index: Initially displayed mass-axis index.
        :type initial_index: int
        :return: Interactive widget container.
        :rtype: Any
        """
        import ipywidgets as widgets
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt

        slider = widgets.IntSlider(
            value=initial_index,
            min=0,
            max=len(self.mass_axis) - 1,
            step=1,
            description="m/z index",
            continuous_update=False,
        )
        output = widgets.Output()

        def refresh(change: Mapping[str, Any] | None = None) -> None:
            """Replace the current figure instead of accumulating outputs."""
            with output:
                clear_output(wait=True)
                figure, _ = self.plot(slider.value)
                display(figure)
                plt.close(figure)

        slider.observe(refresh, names="value")
        refresh()
        return widgets.VBox((slider, output))
