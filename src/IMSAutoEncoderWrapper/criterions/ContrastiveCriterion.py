
# python 
import copy

# numerical
import numpy as np

# ML library 
import torch
import torch.nn.functional as F
from .base import IMSABaseAutoEncoderCriterion
from ..dataset import IMSPyTorchDataset

## functions for find peaks and their envelopes 
from scipy.signal import find_peaks, peak_widths

class ContrastiveCriterion(IMSABaseAutoEncoderCriterion):
    r"""
    Composite loss function for Ion Mobility Spectrometry (IMS) contrastive learning.
    
    This criterion combines InfoNCE contrastive loss with architectural 
    regularization and reconstruction errors.

    **1. Contrastive Loss (InfoNCE - Comparison)**
    The core objective is to maximize the cosine similarity between an original 
    spectrum and its augmented (noisy) version, while minimizing it against all 
    other spectra in the batch.
    
    The calculation follows the formula:
    
    .. math::
       \mathcal{L}_{cont} = -\log \frac{\exp(\text{sim}(z, \bar{z})/\tau)}{\sum_{k=1, k \neq i}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}

    Where:
    * :math:`x_i` is the original spectral value for pixel :math:`i`, and :math:`\bar{x}_i` is the spectral value with applied noise.
    * :math:`z_i, \bar{z}_i` are the normalized latent representations of :math:`x_i, \bar{x}_i`.
    * :math:`\text{sim}(\cdot, \cdot)` is the cosine similarity calculated across a concatenated matrix of size :math:`2N \times 2N`.
    * :math:`\tau` is the temperature hyperparameter (``self.temperature``).

    **2. Variance Regularization (std_loss)**
    Prevents "dimensional collapse" where all latent vectors become identical or 
    inactive. It forces the standard deviation of each latent dimension across the 
    batch towards 1.
    
    .. math::
       \mathcal{L}_{std} = \frac{1}{d} \sum_{k=1}^{d} (\sigma(h_{\cdot, k}) - 1)^2

    Where :math:`h` represents the ``representations`` tensor, and :math:`\sigma` is the 
    standard deviation along the batch dimension (dim=0).

    **3. Mean Regularization (mean_loss)**
    Encourages a zero-centered latent space distribution, which stabilizes the 
    decoder during reconstruction.
    
    .. math::
       \mathcal{L}_{mean} = \left( \frac{1}{N \cdot d} \sum_{i=1}^{N} \sum_{k=1}^{d} h_{ik} \right)^2

    Calculated as the squared mean of all latent features across the batch.

    **4. Reconstruction Loss (decoder_loss)**
    Measures the Root Mean Squared Error (RMSE) between the original spectra 
    (and its noisy counterpart) and their reconstructions (``x_hat``).
    """
    def __init__(self, 
        temperature: float = 2.0,
        max_peaks_per_spectrum: int = 2):
        """
        Initializes the criterion with hardware and scaling settings.

        :param device: The torch device (CPU/GPU) where tensors are processed.
        :type device: torch.device
        :param temperature: Scaling factor for cosine similarity; higher values 
                            create a "softer" probability distribution.
        :type temperature: float, optional
        """
        super().__init__()
        self.temperature = temperature
        # default 
        self.REQUIRED_SETUP = [
            {
                'func': self.precompute_peak_bank, 
                'args': {'max_peaks_per_spectrum': max_peaks_per_spectrum}
            }
        ]
    

    def forward(self, batch_idx, batch_data, model, dataloader, device):
        # Unpack spatial indices and spectral tensors from the dataset
        spatial_indices, x1_raw = batch_data
        
        # Redundant 
        self.device = device
        x1 = x1_raw.to(self.device)  

        # Create noise sample 
        peak_bank = getattr(dataloader.dataset, 'peak_bank', None)
        x2 = self.apply_noise(x1, dataloader.dataset, peak_bank)

        # Embed both vectors
        z1, x1_hat = model(x1)
        z2, x2_hat = model(x2)

        # dynamically download current batch size and adjust size (last batch have have fewer pixels)
        actual_batch_size = z1.size(0)
        
        # contrastive loss
        ## cosine similitaries (sim)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        ## extract positive values
        sim_ij = torch.diag(similarity_matrix, actual_batch_size)
        sim_ji = torch.diag(similarity_matrix, -actual_batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)

        ## mask self similarity (diagonal)
        mask = (~torch.eye(actual_batch_size * 2, actual_batch_size * 2, dtype=torch.bool, device=self.device)).float()
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        ## contrastive loss
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        contrastive_loss = torch.sum(all_losses) / (2 * actual_batch_size)

        ## regularization loss
        std_loss = torch.mean((torch.std(representations, dim=0) - 1) ** 2 + 1e-6)  # for 0 error
        mean_loss = torch.mean(torch.mean(representations, dim=1) ** 2)

        # MSE loss - reconstruction
        inputs_all = torch.cat([x1, x2])
        outputs_all = torch.cat([x1_hat, x2_hat])
        decoder_loss = torch.mean(torch.sqrt(torch.mean((inputs_all - outputs_all) ** 2, dim=1)))

        # total loss
        total_loss = contrastive_loss * 1e-2 + std_loss * 1e-3 + mean_loss * 1e-3 + decoder_loss

         # loss results
        loss_dict = {
            'contrastive_loss': contrastive_loss.item(),
            'std_loss': std_loss.item(),
            'mean_loss': mean_loss.item(),
            'decoder_loss': decoder_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    # ---------------------
    # helpers
    # ---------------------

    @staticmethod
    def apply_noise(
        vec: torch.Tensor, 
        IMSDataset: IMSPyTorchDataset, 
        PeakBank) -> torch.Tensor:
        """
        Augments the input batch by injecting realistic biological/chemical noise.

        This method adds "foreign" peaks sampled from 
        other spectra in the dataset. This ensures that the noise represents 
        potential molecular interference present in the sample, forcing the 
        model to distinguish between invariant chemical signatures and random 
        molecular fluctuations.

        **Mechanism:**
        1. **Peak Detection**: Uses a fast GPU-based local maxima detection to 
           identify peaks in the current batch (comparing values to neighbors 
           and the global mean).
        2. **Dynamic Scaling**: Calculates the number of noise peaks to add based 
           on the current spectrum's complexity (clamped at 5% of existing peaks).
        3. **Bank Injection**: Randomly selects peak envelopes from the 
           ``PeakBank`` and injects them at their original m/z positions directly 
           on the GPU to avoid memory transfer overhead.
        4. **Consistency**: Re-applies normalization (e.g., TIC) to maintain data 
           integrity after augmentation.

        :param vec: The input batch of spectral tensors.
        :type vec: torch.Tensor
        :param IMSDataset: The dataset object containing normalization metadata.
        :type IMSDataset: IMSPyTorchDataset
        :param PeakBank: A list of precomputed peak envelopes (start, end, values).
        :type PeakBank: list
        :returns: A noisy version of the input batch.
        :rtype: torch.Tensor
        """
        # local names
        dataset = IMSDataset
        # obtain vector and params 
        norm_type = dataset.img.normalization
        device = vec.device
        noisy_batch = vec.clone()
        
        # Check if bank exists, if not, the function returns original (failsafe)
        if not hasattr(dataset, 'peak_bank'):
            return noisy_batch

        # Identify peaks in the current batch to determine noise scale
        ## We use a fast GPU-based local maxima detection instead of scipy
        shifted_left = torch.cat([vec[:, 1:], vec[:, -1:]], dim=1)
        shifted_right = torch.cat([vec[:, :1], vec[:, :-1]], dim=1)
        mean_vals = vec.mean(dim=1, keepdim=True)
        
        ## Peak is higher than neighbors and higher than mean
        peak_mask = (vec > shifted_left) & (vec > shifted_right) & (vec > mean_vals)
        num_peaks_in_batch = peak_mask.sum(dim=1)
        
        # Determine number of foreign peaks to add (up to 5%, at least 1)
        num_peaks_to_add = torch.clamp((num_peaks_in_batch * 0.05).int(), min=1)
        max_iterations = num_peaks_to_add.max().item()

        # Iterate through the maximum required additions
        for p_step in range(max_iterations):
            ## Mask for samples that still need peaks added
            active_mask = (num_peaks_to_add > p_step).nonzero(as_tuple=True)[0]
            if len(active_mask) == 0:
                break
                
            ## Randomly select peaks from the precomputed bank for the active batch
            rand_bank_idxs = torch.randint(0, len(PeakBank), (len(active_mask),))
            
            for i, batch_idx in enumerate(active_mask):
                start, end, peak_envelope = PeakBank[rand_bank_idxs[i]]
                ## Add the peak envelope directly on GPU
                noisy_batch[batch_idx, start:end] += peak_envelope.to(device)

        # Re-normalize - data consistency 
        if norm_type == 'TIC':
            ### Total Ion Count normalization: ensure sum of intensities is constant
            noisy_tic = torch.sum(noisy_batch, dim=1, keepdim=True)
            noisy_batch = noisy_batch / noisy_tic.clamp(min=1e-12)

        #TODO apply other normalization types
            
        return noisy_batch

    @staticmethod
    def precompute_peak_bank(
        dataset: IMSPyTorchDataset, 
        max_peaks_per_spectrum: int = 2):
        """
        Pre-identifies and extracts peak envelopes across the entire dataset to 
        build a noise reference library.

        This one-time setup step scans the dataset to find high-intensity peaks 
        using local maxima and width estimation. These envelopes are stored 
        as small tensors to be reused during training, providing a diverse 
        pool of "biological noise" without the need for real-time 
        peak-picking.

        **Process:**
        1. **Identification**: Uses ``scipy.signal.find_peaks`` to locate 
           significant peaks relative to the mean intensity.
        2. **Envelope Extraction**: Calculates the width of each peak at 80% 
           relative height to capture the full spectral profile.
        3. **Memory Optimization**: Limits the number of stored peaks per 
           spectrum (``max_peaks_per_spectrum``) to prevent excessive 
           RAM usage.

        :param dataset: The dataset to scan for peaks.
        :type dataset: IMSPyTorchDataset
        :param max_peaks_per_spectrum: Maximum number of peaks to extract from 
                                       a single spectrum.
        :type max_peaks_per_spectrum: int
        :returns: A list of tuples containing (start_index, end_index, peak_values).
        :rtype: list
        """
        peak_bank = []

        if hasattr(dataset, 'peak_bank') and dataset.peak_bank:
            print("PeakBank already exists in dataset. Skipping precomputation.")
            return
        
        print(f"Building noise bank (max {max_peaks_per_spectrum} peaks per spectrum)...")
        
        for i in range(len(dataset)):
            ## Load spectrum to CPU
            _, spectrum_np = dataset[i]
            spectrum_np = spectrum_np.numpy()
            
            ## Find peaks (scipy library) 
            peaks, _ = find_peaks(spectrum_np, height=np.mean(spectrum_np))
            
            if len(peaks) > 0:
                ## Limit the number of peaks stored per spectrum to save memory
                selected_peaks = np.random.choice(
                    peaks, 
                    size=min(len(peaks), max_peaks_per_spectrum), 
                    replace=False
                )
                
                for p_idx in selected_peaks:
                    ### Calculate envelope width using original parameters
                    widths, _, left_ips, right_ips = peak_widths(
                        spectrum_np, [p_idx], rel_height=0.8
                    )
                    
                    start = int(left_ips[0])
                    end = int(right_ips[0]) + 1
                    
                    ### Store as small torch float16/32 tensors to save space
                    peak_vals = torch.from_numpy(spectrum_np[start:end]).float()
                    peak_bank.append((start, end, peak_vals))
                    
        ## Attach the bank to the dataset object
        dataset.peak_bank = peak_bank
        print(f"PeakBank created with {len(peak_bank)} total noise samples.")
        