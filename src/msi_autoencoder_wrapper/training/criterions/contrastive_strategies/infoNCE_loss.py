import random
from typing import Any, Dict, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, peak_widths

from ..criterions_manager import CriterionsManager
from ..base_criterion import MSIBaseCriterion
from ....models.datasets.strategies.base_dataset import MSIBaseDataset


@CriterionsManager.register_criterion("InfoNCELoss")
class MSIInfoNCELoss(MSIBaseCriterion):
    """
    InfoNCE Contrastive Loss strategy tailored for MSI chemical spatial mapping alignment.

    This block computes multi-class contrastive step updates over original profile projections
    and augmented instances created on-the-fly using localized chemical noise banks.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        max_peaks_per_spectrum: int = 5,
        noise_level: float = 0.05
    ) -> None:
        """
        Configures hyperparameter states for contrastive mapping evaluations.

        :param temperature: Softmax scaling factor balancing gradient scale properties.
        :type temperature: float
        :param max_peaks_per_spectrum: Limit tracking maximum extracted noise profiles stored per spectrum.
        :type max_peaks_per_spectrum: int
        :param noise_level: Amplitude modifier tracking injected random noise transformations.
        :type noise_level: float
        """
        super().__init__()
        self._config = {
            "temperature": temperature,
            "max_peaks_per_spectrum": max_peaks_per_spectrum,
            "noise_level": noise_level
        }
        self.temperature = temperature
        self.max_peaks_per_spectrum = max_peaks_per_spectrum
        self.noise_level = noise_level
        
        # Internal memory allocation for peak envelopes storage
        self.peak_bank: List[Tuple[int, int, torch.Tensor]] = []

    def REQUIRED_SETUP(self, dataset: MSIBaseDataset) -> None:
        """
        Scans the active dataset to assemble the global localized peak noise profile bank.
        """
        print(f"[InfoNCE Setup] Extracting peak envelopes to generate baseline noise banks...")
        temp_bank: List[Tuple[int, int, torch.Tensor]] = []
        
        for i in range(len(dataset)):
            _, spectrum_tensor = dataset[i]
            spectrum_np = spectrum_tensor.numpy()
            
            # Detect peaks utilizing mean intensity threshold footprints
            peaks, _ = find_peaks(spectrum_np, height=np.mean(spectrum_np))
            
            if len(peaks) > 0:
                selected_peaks = np.random.choice(
                    peaks,
                    size=min(len(peaks), self.max_peaks_per_spectrum),
                    replace=False
                )
                
                for p_idx in selected_peaks:
                    # Calculate peak boundary base coordinate envelopes using 80% relative depth metrics
                    widths, _, left_ips, right_ips = peak_widths(
                        spectrum_np, [p_idx], rel_height=0.8
                    )
                    start = int(left_ips[0])
                    end = int(right_ips[0]) + 1
                    
                    peak_vals = torch.from_numpy(spectrum_np[start:end]).float()
                    temp_bank.append((start, end, peak_vals))
                    
        self.peak_bank = temp_bank
        print(f"[InfoNCE Setup] Noise bank compiled successfully. Total stored signatures: {len(self.peak_bank)}.")

    @property
    def requires_reconstruction(self) -> bool:
        """Does not operate on reconstructed spectrum channel maps."""
        return False

    @property
    def requires_projection(self) -> bool:
        """Requires active contrastive projector head representations tensors."""
        return True

    def _apply_noise_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Injects localized modifications into input vectors leveraging cached bank entries."""
        if not self.peak_bank:
            return x
            
        x_noise = x.clone()
        batch_size, grid_depth = x_noise.shape
        
        for b in range(batch_size):
            # Select a random peak signature envelope configuration context from memory arrays
            start, end, peak_vals = random.choice(self.peak_bank)
            width = end - start
            
            if width < grid_depth:
                # Align placement indices to apply localized noise alterations safely
                target_start = random.randint(0, grid_depth - width)
                target_end = target_start + width
                
                # Apply balanced transformation step scaling properties
                x_noise[b, target_start:target_end] += peak_vals.to(x.device) * self.noise_level
                
        return x_noise

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Evaluates the InfoNCE contrastive step score using symmetric similarity tracking matrices.
        """
        if "projection" not in model_outputs:
            raise KeyError("The contrastive 'projection' tensor is missing from forward pass output records maps.")

        _, original_spectra = batch_data
        batch_size = original_spectra.shape[0]

        # 1. Execute dynamic localized data augmentation step using the compiled noise bank channels
        augmented_spectra = self._apply_noise_augmentation(original_spectra)

        # 2. Extract corresponding model representations using the active backpropagation execution graph
        # This forward path context lookup leverages the caller framework environment parameters
        model_ref = kwargs.get("model_reference")
        if model_ref is None:
            raise RuntimeError("Contrastive execution pass requires explicit model_reference tracking keywords bindings.")
            
        # Extract contrastive vector targets for both matched states simultaneously
        z_orig = model_outputs["projection"]
        _, _, z_aug = model_ref.forward_optimized(augmented_spectra, requires_reconstruction=False, requires_projection=True)

        # 3. Apply standard L2-normalization transformations to format cosine calculation spaces
        z_orig_norm = F.normalize(z_orig, dim=1)
        z_aug_norm = F.normalize(z_aug, dim=1)

        # 4. Construct complete concatenated dual projection layout matrix tracking elements [2N, Projection_Dim]
        representations = torch.cat([z_orig_norm, z_aug_norm], dim=0)

        # 5. Compute the massive symmetrical cosine similarity matrix tracking cross intersections [2N, 2N]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # 6. Extract positive match locations using structured diagonal grid masking indices shifts
        diag_pos_1 = torch.diag(similarity_matrix, batch_size)
        diag_pos_2 = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([diag_pos_1, diag_pos_2], dim=0).view(2 * batch_size, 1)

        # 7. Isolate negative match locations by pruning identity reflection cells from calculation operations
        mask = torch.eye(2 * batch_size, device=original_spectra.device, dtype=torch.bool)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

        # 8. Unify elements log representations and extract cumulative multi-class categorical cross-entropy loss
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, device=original_spectra.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)