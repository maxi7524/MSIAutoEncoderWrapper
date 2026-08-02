"""
InfoNCE Contrastive Loss strategy tailored for MSI chemical spatial mapping alignment.
"""

from typing import Any, Dict, Tuple
import numpy as np
import torch
from scipy.signal import find_peaks, peak_widths

from ...autoencoder_base_criterions import MSIContrastiveCriterion
from ...criterions_manager import CriterionsManager
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.logger import get_custom_logger
from .....utils.exceptions import raise_incompatible_interface_error
from .....data import SpectrumBatch
from .....metrics import info_nce

# Logger initialization
logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion("autoencoder", "contrastive", "InfoNCELoss")
class MSIInfoNCELoss(MSIContrastiveCriterion):
    """
    InfoNCE Contrastive Loss using automated chemical peak profiling noise caches.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        noise_level: float = 0.05,
        max_peaks_per_spectrum: int = 2,
        peak_sample_size: int = 1000,
        peak_sample_seed: int = 0,
        max_noise_peaks: int = 8,
    ) -> None:
        """
        Configures hyperparameter states for contrastive mapping evaluations.

        :param temperature: Softmax scaling factor balancing gradient scale properties.
        :type temperature: float
        :param noise_level: Amplitude modifier tracking injected random noise transformations.
        :type noise_level: float
        :param max_peaks_per_spectrum: Peak envelopes retained per sampled spectrum.
        :type max_peaks_per_spectrum: int
        :param peak_sample_size: Maximum dataset spectra scanned for the peak bank.
        :type peak_sample_size: int
        :param peak_sample_seed: Reproducible dataset sampling seed.
        :type peak_sample_seed: int
        :param max_noise_peaks: Maximum foreign envelopes added to one spectrum.
        :type max_noise_peaks: int
        """
        super().__init__()
        self._config = {
            "temperature": temperature,
            "noise_level": noise_level,
            "max_peaks_per_spectrum": max_peaks_per_spectrum,
            "peak_sample_size": peak_sample_size,
            "peak_sample_seed": peak_sample_seed,
            "max_noise_peaks": max_noise_peaks,
        }
        self.temperature = temperature
        self.noise_level = noise_level
        self.max_peaks_per_spectrum = max_peaks_per_spectrum
        self.peak_sample_size = peak_sample_size
        self.peak_sample_seed = peak_sample_seed
        self.max_noise_peaks = max_noise_peaks

    def on_phase_start(self, model: torch.nn.Module, dataset: MSIBaseDataset, transient_cache: Dict[str, Any]) -> None:
        """
        Pre-computes a steady noise mask across dataset spectra profiles to cache peak positions.
        """
        # Heading 1 (Chemical Peak Profiles Extraction Pass)
        if "chemical_peak_bank" in transient_cache:
            logger.info("Reusing the chemical peak-envelope bank from transient cache.")
            return

        logger.info("Pre-calculating reusable chemical peak envelopes.")
        sample_size = min(len(dataset), self.peak_sample_size)
        peak_bank = []
        sample_indices = np.random.default_rng(self.peak_sample_seed).choice(
            len(dataset),
            size=sample_size,
            replace=False,
        )

        for idx in sample_indices:
            sample = dataset[idx]
            spectrum_tensor = sample[1]
            spectrum = spectrum_tensor.detach().cpu().numpy()
            peaks, properties = find_peaks(
                spectrum,
                prominence=max(float(np.mean(spectrum)), 0.0),
            )
            if peaks.size == 0:
                continue
            prominences = properties.get("prominences", np.ones_like(peaks))
            selected = peaks[np.argsort(prominences)[-self.max_peaks_per_spectrum :]]
            _, _, left_ips, right_ips = peak_widths(
                spectrum,
                selected,
                rel_height=0.8,
            )
            for left, right in zip(left_ips, right_ips):
                start = max(0, int(np.floor(left)))
                stop = min(len(spectrum), int(np.ceil(right)) + 1)
                if stop > start:
                    peak_bank.append(
                        (start, stop, torch.as_tensor(spectrum[start:stop]).float())
                    )

        transient_cache["chemical_peak_bank"] = peak_bank
        logger.info(
            "Chemical profiling complete. Cached %s peak envelopes.",
            len(peak_bank),
        )

    def on_batch_start(self, batch_data: Tuple[torch.Tensor, ...], transient_cache: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Dynamic online augmentation step doubling the batch matrix size (2N) via targeted chemical noise.
        """
        # Heading 1 (On-the-fly Batch Chemical Noise Augmentation Pass)
        if isinstance(batch_data, SpectrumBatch):
            original_spectra = batch_data.spectra
        else:
            _, original_spectra = batch_data
        augmented_spectra = original_spectra.clone()
        peak_bank = transient_cache.get("chemical_peak_bank", [])

        if peak_bank:
            peak_mask = (
                (original_spectra[:, 1:-1] > original_spectra[:, :-2])
                & (original_spectra[:, 1:-1] > original_spectra[:, 2:])
                & (original_spectra[:, 1:-1] > original_spectra.mean(dim=1, keepdim=True))
            )
            additions = torch.clamp(
                (peak_mask.sum(dim=1).float() * self.noise_level).ceil().long(),
                min=1,
                max=self.max_noise_peaks,
            )
            for batch_index, addition_count in enumerate(additions.tolist()):
                selected = torch.randint(len(peak_bank), (addition_count,)).tolist()
                for bank_index in selected:
                    start, stop, values = peak_bank[bank_index]
                    augmented_spectra[batch_index, start:stop] += values.to(
                        augmented_spectra.device,
                        dtype=augmented_spectra.dtype,
                    )

        ## Stack original and augmented spectra down the batch dimension to pass 2N features into forward pass
        if isinstance(batch_data, SpectrumBatch):
            return batch_data.with_view("contrastive", augmented_spectra)
        spatial_indices = batch_data[0]
        combined_spectra = torch.cat([original_spectra, augmented_spectra], dim=0)
        combined_indices = torch.cat([spatial_indices, spatial_indices], dim=0)
        return (combined_indices, combined_spectra)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Evaluates multi-class categorical cross-entropy scores across contrastive vector splits.
        """
        # Heading 1 (Contrastive InfoNCE Metric Alignment Pass)
        if "projection" not in model_outputs:
            raise_incompatible_interface_error(
                context_name="InfoNCELoss",
                message="Model outputs must contain a 'projection' tensor.",
            )

        projection = model_outputs["projection"]
        batch_size = projection.shape[0] // 2

        ## Extract original and augmented latent halves from the 2N forwarded outputs matrix
        z_orig = projection[:batch_size]
        z_aug = projection[batch_size:]

        ## Apply standard L2-normalization transformations to format cosine calculation spaces
        return info_nce(z_orig, z_aug, temperature=self.temperature).mean()
