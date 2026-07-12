"""
InfoNCE Contrastive Loss strategy tailored for MSI chemical spatial mapping alignment.
"""

from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from ...base_criterion import MSIBaseCriterion
from ...criterions_manager import CriterionsManager
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion("autoencoder", "InfoNCELoss")
class MSIInfoNCELoss(MSIBaseCriterion):
    """
    InfoNCE Contrastive Loss using automated chemical peak profiling noise caches.
    """

    def __init__(self, temperature: float = 0.07, noise_level: float = 0.05) -> None:
        """
        Configures hyperparameter states for contrastive mapping evaluations.

        :param temperature: Softmax scaling factor balancing gradient scale properties.
        :type temperature: float
        :param noise_level: Amplitude modifier tracking injected random noise transformations.
        :type noise_level: float
        """
        super().__init__()
        self._config = {"temperature": temperature, "noise_level": noise_level}
        self.temperature = temperature
        self.noise_level = noise_level

    def on_phase_start(self, model: torch.nn.Module, dataset: MSIBaseDataset, transient_cache: Dict[str, Any]) -> None:
        """
        Pre-computes a steady noise mask across dataset spectra profiles to cache peak positions.
        """
        # Heading 1 (Chemical Peak Profiles Extraction Pass)
        if "chemical_noise_mask" in transient_cache:
            logger.info("Reusing initialized chemical noise mask channels from transient cache.")
            return

        logger.info("Pre-calculating chemical peak locations across dataset sample slices.")
        sample_size = min(len(dataset), 100)
        discovered_peaks = set()

        for idx in range(sample_size):
            try:
                _, spectrum_tensor = dataset[idx]
                peaks, _ = find_peaks(spectrum_tensor.numpy(), prominence=0.1)
                discovered_peaks.update(peaks)
            except Exception:
                continue

        transient_cache["chemical_noise_mask"] = list(discovered_peaks)
        logger.info("Chemical profiling complete. Cached %s distinct noise channels.", len(transient_cache["chemical_noise_mask"]))

    def on_batch_start(self, batch_data: Tuple[torch.Tensor, ...], transient_cache: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Dynamic online augmentation step doubling the batch matrix size (2N) via targeted chemical noise.
        """
        # Heading 1 (On-the-fly Batch Chemical Noise Augmentation Pass)
        spatial_indices, original_spectra = batch_data
        augmented_spectra = original_spectra.clone()
        noise_channels = transient_cache.get("chemical_noise_mask", [])

        if noise_channels:
            ### Generate Gaussian jitter noise targeted onto verified peak index channels
            noise_tensor = torch.zeros_like(augmented_spectra)
            noise_tensor[:, noise_channels] = torch.randn((augmented_spectra.shape[0], len(noise_channels))) * self.noise_level
            augmented_spectra += noise_tensor

        ## Stack original and augmented spectra down the batch dimension to pass 2N features into forward pass
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
            logger.error("Evaluation aborted: 'projection' key missing from model outputs.")
            raise KeyError("InfoNCELoss requires 'projection' field in model outputs.")

        projection = model_outputs["projection"]
        batch_size = projection.shape[0] // 2

        ## Extract original and augmented latent halves from the 2N forwarded outputs matrix
        z_orig = projection[:batch_size]
        z_aug = projection[batch_size:]

        ## Apply standard L2-normalization transformations to format cosine calculation spaces
        z_orig_norm = F.normalize(z_orig, dim=1)
        z_aug_norm = F.normalize(z_aug, dim=1)

        ## Construct complete concatenated dual projection layout matrix tracking elements [2N, Projection_Dim]
        representations = torch.cat([z_orig_norm, z_aug_norm], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        ## Extract positive match locations using structured diagonal grid masking indices shifts
        diag_pos_1 = torch.diag(similarity_matrix, batch_size)
        diag_pos_2 = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([diag_pos_1, diag_pos_2], dim=0).view(2 * batch_size, 1)

        ## Isolate negative match locations by pruning identity reflection cells from calculation operations
        mask = torch.eye(2 * batch_size, device=projection.device, dtype=torch.bool)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

        ## Unify elements log representations and extract cumulative cross-entropy scores
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, device=projection.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)