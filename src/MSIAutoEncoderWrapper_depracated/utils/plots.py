import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

class MSIModelVisualizer:
    """
    Mixin class to extend MSIContrastiveModel with visualization capabilities.
    """
    
    def plot_epoch_losses(self, title="Training Losses", xlabel="Epochs", ylabel="Loss", figsize_per_plot=(4, 4)):
        """
        Plots training history from the model.
        """
        # W Twoim modelu historia to self._history (lista słowników)
        if not hasattr(self, '_history') or not self._history:
            print("[UTILS - plots]: No training history found.")
            return

        # 1. Konwersja do DataFrame dla łatwej manipulacji
        df = pd.DataFrame(self._history)
        
        # 2. Usuwamy kolumny, których nie chcemy na wykresie (np. 'epoch')
        cols_to_plot = [c for c in df.columns if c.lower() != 'epoch']
        n_metrics = len(cols_to_plot)

        # 3. Dynamiczne ustawienie siatki (grid)
        cols = min(3, n_metrics) # max 3 wykresy w rzędzie
        rows = int(np.ceil(n_metrics / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]))
        
        # Spłaszczenie osi, żeby łatwo po nich iterować
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col_name in enumerate(cols_to_plot):
            ax = axes[idx]
            ax.plot(df[col_name], label=col_name, linewidth=2, color=f'C{idx}')
            ax.set_title(col_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # 4. Ukrycie pustych osi, jeśli metryk jest mniej niż pól w siatce
        for i in range(idx + 1, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


        # # Konwersja listy słowników na słownik list (dla kompatybilności z kodem plota)
        # metrics_dict = {}
        # keys_to_plot = [k for k in self._history[0].keys() if k != 'epoch']
        
        # for key in keys_to_plot:
        #     metrics_dict[key] = [epoch[key] for epoch in self._history]

        # n_metrics = len(metrics_dict)
        # cols = min(4, n_metrics)
        # rows = int(np.ceil(n_metrics / cols))
        
        # fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]))
        
        # if n_metrics == 1:
        #     axes = np.array([axes])
        # axes = axes.flatten()

        # for idx, (metric_name, values) in enumerate(metrics_dict.items()):
        #     axes[idx].plot(values, label=metric_name, linewidth=2, marker='o', markersize=3)
        #     axes[idx].set_title(metric_name.replace('_', ' ').title())
        #     axes[idx].set_xlabel(xlabel)
        #     axes[idx].set_ylabel(ylabel)
        #     axes[idx].grid(True, linestyle='--', alpha=0.7)
        #     axes[idx].legend()

        # # Ukryj puste wykresy
        # for idx in range(n_metrics, len(axes)):
        #     axes[idx].set_visible(False)

        # plt.suptitle(title, fontsize=16)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()

    def plot_reconstruction(self, data_batch, apply_noise_fn=None):
        """
        Visualizes Input, (Noised), Latent space and Reconstruction.
        """
        if self._model is None:
            raise ValueError("Model is not initialized.")

        self._model.eval()
        device = self._device

        # Przygotowanie tensora
        if isinstance(data_batch, np.ndarray):
            X_tensor = torch.from_numpy(data_batch).float()
        elif isinstance(data_batch, torch.Tensor):
            X_tensor = data_batch.float()
        else:
            # Zakładamy, że to pojedynczy indeks z Loadera
            X_tensor = torch.tensor(data_batch).float()

        X_tensor = X_tensor.to(device)
        
        # Dopasowanie wymiarów (Batch, Features)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)

        plots_to_show = [("Original Spectrum", X_tensor.detach().cpu().numpy().squeeze(), 'r')]

        with torch.no_grad():
            if apply_noise_fn is not None:
                noised_tensor = apply_noise_fn(X_tensor)
                plots_to_show.append(("Noised Spectrum", noised_tensor.detach().cpu().numpy().squeeze(), 'gray'))
                model_input = noised_tensor
            else:
                model_input = X_tensor

            # Forward pass: Twój model zwraca (z_norm, x_hat)
            z_norm, x_hat = self._model(model_input)
            
            plots_to_show.append(("Latent Space (z)", z_norm.cpu().numpy().squeeze(), 'g'))
            plots_to_show.append(("Reconstruction", x_hat.cpu().numpy().squeeze(), 'b'))

        # Rysowanie
        num_plots = len(plots_to_show)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        
        if num_plots == 1:
            axes = [axes]

        for ax, (title, data, color) in zip(axes, plots_to_show):
            ax.plot(data, color=color, alpha=0.7)
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()