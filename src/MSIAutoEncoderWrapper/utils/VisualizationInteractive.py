"""
visualization/interactive.py
----------------------------
Interactive tools for exploring IMS data.
Uses ipywidgets to drive the m2aia reader directly using its native axis.
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from IPython.display import display
import m2aia as m2

class M2AIAExplorer:
    """
    Interaktywny eksplorator obrazów jonowych dla M2AIA (pym2aia).
    """

    def __init__(self, reader):
        self.reader = reader
        # Pobieramy oś m/z bezpośrednio z readera M2AIA
        self.mz_axis = reader.GetXAxis()
        
        self.mz_min = self.mz_axis.min()
        self.mz_max = self.mz_axis.max()

        self._common_vmax_cache = {}

    def plot(self, mz_idx_init: int = 0, tol_init: float = 0.1):

        def _view(change=None):
            mz_idx = mz_slider.value
            tol = tol_slider.value
            max_percentile = percentile_slider.value
            use_common_scale = common_scale_checkbox.value

            mz_val = self.mz_axis[mz_idx]
            
            # M2AIA: GetArray zwraca 3D (z, y, x). Squeezujemy, żeby dostać 2D
            # Format: reader.GetArray(mz, tolerance)
            img_3d = self.reader.GetArray(mz_val, tol)
            img = np.squeeze(img_3d)

            output.clear_output(wait=True)

            with output:
                fig, ax = plt.subplots(figsize=(8, 6))

                if use_common_scale:
                    # Obliczanie wspólnej skali dla aktualnego kwantyla (jeśli nie ma w cache)
                    # Uwaga: W M2AIA obliczenie tego dla całego zbioru może trwać.
                    # Tutaj liczymy vmax dla aktualnie wyświetlanego obrazu jako punkt odniesienia, 
                    # lub możesz to zastąpić stałą wartością.
                    if max_percentile not in self._common_vmax_cache:
                        # Przybliżenie: bierzemy max z obecnego obrazu lub całego zestawu 
                        # (M2AIA nie ma apply_over, więc używamy np.percentile na obecnym img)
                        vmax = np.percentile(img, max_percentile) if np.max(img) > 0 else 1e-8
                        self._common_vmax_cache[max_percentile] = vmax
                    
                    vmax = self._common_vmax_cache[max_percentile]
                else:
                    vmax = np.percentile(img, max_percentile) if np.max(img) > 0 else 1e-8

                im = ax.imshow(
                    img,
                    cmap="inferno",
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest"
                )

                ax.set_title(
                    f"m/z = {mz_val:.4f} ± {tol} Da\nMax Int: {np.max(img):.2e}",
                    fontsize=12
                )
                ax.axis("off")

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Intensity")

                plt.tight_layout()
                plt.show()

        # -------- Widgets --------

        # Tworzymy listę opcji dla slidera (tekst, indeks)
        # Przy bardzo dużej liczbie m/z SelectionSlider może zwolnić - wtedy lepiej użyć IntSlider
        mz_options = [(f"{mz:.3f}", i) for i, mz in enumerate(self.mz_axis)]

        mz_slider = widgets.SelectionSlider(
            options=mz_options,
            value=mz_idx_init,
            description='m/z Center',
            continuous_update=False,
            layout=Layout(width='100%')
        )

        tol_slider = widgets.FloatSlider(
            value=tol_init,
            min=0.001,
            max=2.0,
            step=0.005,
            description='Tol (+/-)',
            continuous_update=False,
            layout=Layout(width='100%')
        )

        percentile_slider = widgets.FloatSlider(
            value=100.0,
            min=80.0,
            max=100.0,
            step=0.1,
            description='Upper Quantile',
            continuous_update=False,
            readout_format='.1f',
            layout=Layout(width='70%')
        )

        common_scale_checkbox = widgets.Checkbox(
            value=False,
            description='Common Scale',
            indent=False,
            layout=Layout(width='30%')
        )

        # -------- Layout --------

        quantile_row = HBox(
            [percentile_slider, common_scale_checkbox],
            layout=Layout(width='100%', align_items='center')
        )

        controls = VBox(
            [mz_slider, tol_slider, quantile_row],
            layout=Layout(width='100%')
        )

        output = widgets.Output(layout=Layout(width='100%'))

        container = VBox(
            [controls, output],
            layout=Layout(width='100%')
        )

        # Trigger events
        mz_slider.observe(_view, names='value')
        tol_slider.observe(_view, names='value')
        percentile_slider.observe(_view, names='value')
        common_scale_checkbox.observe(_view, names='value')

        display(container)
        _view()

# --- Przykład użycia w Notebooku ---
# import m2aia as m2
# reader = m2.ImzMLReader("sciezka/do/pliku.imzML", normalization=m2.m2NormalizationRMS)
# explorer = M2AIAExplorer(reader)
# explorer.plot()



class M2AIAMultiExplorer:
    """
    Interactive Multi-Image Explorer for M2AIA.
    Allows side-by-side comparison of multiple imzML files with synchronized controls.
    """

    def __init__(self, readers, labels=None):
        """
        :param readers: List of m2.ImzMLReader objects.
        :param labels: List of strings for titles (optional).
        """
        self.readers = readers
        self.labels = labels if labels else [f"Sample {i+1}" for i in range(len(readers))]
        
        # We assume all files have a similar m/z range; using the first one as reference
        self.mz_axis = readers[0].GetXAxis()
        
        # Pre-calculate layout based on number of readers
        self.num_samples = len(self.readers)
        self.ncols = min(3, self.num_samples)
        self.nrows = int(np.ceil(self.num_samples / self.ncols))

    def plot(self, mz_idx_init: int = 0, tol_init: float = 0.1):
        # UI Output area
        output = widgets.Output()

        def _update_view(change=None):
            mz_idx = mz_slider.value
            tol = tol_slider.value
            max_percentile = percentile_slider.value
            use_common_scale = common_scale_checkbox.value

            mz_val = self.mz_axis[mz_idx]
            
            # 1. Fetch all images
            images = []
            for r in self.readers:
                img_3d = r.GetArray(mz_val, tol)
                images.append(np.squeeze(img_3d))

            # 2. Determine scaling
            # If Common Scale is on, we find the global max across all current views
            global_vmax = 0
            if use_common_scale:
                all_vals = [np.percentile(img, max_percentile) for img in images if np.max(img) > 0]
                global_vmax = max(all_vals) if all_vals else 1e-8

            output.clear_output(wait=True)

            with output:
                # Dynamic figure size based on grid
                fig, axes = plt.subplots(
                    self.nrows, self.ncols, 
                    figsize=(5 * self.ncols, 4 * self.nrows),
                    constrained_layout=True
                )
                
                # Handle single-plot case (flatten axes)
                if self.num_samples == 1:
                    axes_list = [axes]
                else:
                    axes_list = axes.flatten()

                for i in range(len(axes_list)):
                    ax = axes_list[i]
                    if i < self.num_samples:
                        img = images[i]
                        
                        # Use local or global scaling
                        vmax = global_vmax if use_common_scale else np.percentile(img, max_percentile)
                        if vmax <= 0: vmax = 1e-8

                        im = ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
                        ax.set_title(f"{self.labels[i]}\nMax: {np.max(img):.2e}", fontsize=10)
                        
                        # Add individual colorbars for clarity if not common scale
                        plt.colorbar(im, ax=ax, shrink=0.8, label="Int" if i == self.num_samples - 1 else "")
                    
                    ax.axis("off")

                fig.suptitle(f"m/z = {mz_val:.4f} ± {tol} Da", fontsize=14, fontweight='bold')
                plt.show()

        # -------- UI Widgets Construction --------

        mz_options = [(f"{mz:.3f}", i) for i, mz in enumerate(self.mz_axis)]

        mz_slider = widgets.SelectionSlider(
            options=mz_options,
            value=mz_idx_init,
            description='m/z Center:',
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=Layout(width='98%')
        )

        tol_slider = widgets.FloatSlider(
            value=tol_init, min=0.001, max=1.0, step=0.005,
            description='Tolerance (+/-):',
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=Layout(width='98%')
        )

        percentile_slider = widgets.FloatSlider(
            value=100.0, min=90.0, max=100.0, step=0.1,
            description='Intensity Quantile:',
            continuous_update=False,
            style={'description_width': 'initial'},
            layout=Layout(width='60%')
        )

        common_scale_checkbox = widgets.Checkbox(
            value=True,
            description='Sync Color Scale',
            indent=False,
            layout=Layout(width='30%')
        )

        # -------- Event Listeners --------
        mz_slider.observe(_update_view, 'value')
        tol_slider.observe(_update_view, 'value')
        percentile_slider.observe(_update_view, 'value')
        common_scale_checkbox.observe(_update_view, 'value')

        # -------- Display Layout --------
        controls = VBox([
            mz_slider, 
            tol_slider, 
            HBox([percentile_slider, common_scale_checkbox], layout=Layout(justify_content='space-between'))
        ], layout=Layout(padding='10px', border='1px solid #ddd', margin='0 0 10px 0'))

        display(VBox([controls, output]))
        _update_view()

# --- Example Usage ---
# r1 = m2.ImzMLReader("sample1.imzML")
# r2 = m2.ImzMLReader("sample2.imzML")
# explorer = M2AIAMultiExplorer([r1, r2], labels=["Wild Type", "Knockout"])
# explorer.plot()