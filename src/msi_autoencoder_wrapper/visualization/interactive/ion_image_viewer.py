"""Interactive traversal of aligned ion-image collections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

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
        renderer: Optional[Callable[[float, Mapping[str, np.ndarray]], Any]] = None,
    ) -> None:
        self.mass_axis = np.asarray(mass_axis, dtype=np.float64)
        self.image_provider = image_provider
        self.theme = resolve_theme(theme)
        self.renderer = renderer
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
        if self.renderer is not None:
            return self.renderer(mass, images)
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


class ContinuousIonImageViewer:
    """Display lazily generated ion images selected by a floating-point m/z value.

    Unlike :class:`IonImageViewer`, this viewer exposes physical m/z values directly
    instead of integer positions in a predefined sequence. Generated image collections
    are cached by m/z value for fast revisits during one notebook session.

    :param mz_min: Lower slider boundary.
    :type mz_min: float
    :param mz_max: Upper slider boundary.
    :type mz_max: float
    :param mz_step: Slider increment in m/z units.
    :type mz_step: float
    :param image_provider: Callable returning named spatial arrays for one m/z.
    :type image_provider: Callable[[float], Mapping[str, numpy.ndarray]]
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    """

    def __init__(
        self,
        mz_min: float,
        mz_max: float,
        mz_step: float,
        image_provider: Callable[[float], Mapping[str, np.ndarray]],
        theme: VisualizationTheme | str | None = None,
        renderer: Optional[Callable[[float, Mapping[str, np.ndarray]], Any]] = None,
    ) -> None:
        self.mz_min = float(mz_min)
        self.mz_max = float(mz_max)
        self.mz_step = float(mz_step)
        self.image_provider = image_provider
        self.theme = resolve_theme(theme)
        self.renderer = renderer
        self._image_cache: dict[float, Mapping[str, np.ndarray]] = {}
        if not np.all(np.isfinite((self.mz_min, self.mz_max, self.mz_step))):
            raise ValueError("m/z slider boundaries and step must be finite.")
        if self.mz_min > self.mz_max:
            raise ValueError("mz_min cannot be greater than mz_max.")
        if self.mz_step <= 0.0:
            raise ValueError("mz_step must be positive.")

    def plot(self, mz: float) -> Any:
        """Render the image collection for one physical m/z value.

        :param mz: Mass value inside the configured slider range.
        :type mz: float
        :return: Figure and axes returned by the spatial grid renderer.
        :rtype: Any
        """
        mass = float(mz)
        if not self.mz_min <= mass <= self.mz_max:
            raise ValueError("mz must belong to the configured slider range.")
        cache_key = round(mass, 12)
        if cache_key not in self._image_cache:
            self._image_cache[cache_key] = self.image_provider(mass)
        images = self._image_cache[cache_key]
        if self.renderer is not None:
            return self.renderer(mass, images)
        figure, axes = plot_image_grid(images, theme=self.theme)
        figure.suptitle(f"m/z {mass:.5f}")
        return figure, axes

    def widget(self, initial_mz: Optional[float] = None) -> Any:
        """Return an ipywidgets floating-point m/z slider and reusable output area.

        :param initial_mz: Initially displayed m/z value. Defaults to ``mz_min``.
        :type initial_mz: Optional[float]
        :return: Interactive widget container.
        :rtype: Any
        """
        import ipywidgets as widgets
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt

        initial = self.mz_min if initial_mz is None else float(initial_mz)
        if not self.mz_min <= initial <= self.mz_max:
            raise ValueError("initial_mz must belong to the configured slider range.")
        slider = widgets.FloatSlider(
            value=initial,
            min=self.mz_min,
            max=self.mz_max,
            step=self.mz_step,
            description="m/z",
            continuous_update=False,
            readout_format=".5f",
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
