import numpy as np
from scipy.signal import find_peaks, peak_widths

# Library imports 
## base dataset
from ...datasets.strategies.pixel_dataset import MSIPyTorchDataset


def estimate_max_peak_width(dataset: MSIPyTorchDataset, sample_size: int = 100) -> int:
    """
    Analyzes a random statistical sample of mass spectra to estimate the largest peak envelope width.

    This utility reads raw intensity profiles from the dataset to compute peak base widths
    expressed in grid-x-axis bin units. The maximum detected width serves as an empirical
    guide to automatically scale the kernel receptive field in the initial 1D convolutional layer.

    :param dataset: Bound dataset object providing aligned spectra on the grid-x-axis.
    :type dataset: MSIPyTorchDataset
    :param sample_size: Number of random pixel profiles to analyze, defaults to 100.
    :type sample_size: int
    :return: Suggested asymmetric odd kernel size dimension (minimum value is 3).
    :rtype: int
    """
    total_spectra = len(dataset)
    # Perform non-replacement random sampling across available pixel indices
    indices = np.random.choice(total_spectra, min(sample_size, total_spectra), replace=False)
    
    max_width = 0.0
    
    for idx in indices:
        # Extract binned intensity tensor from the dataset pipeline
        _, spectrum_tensor = dataset[idx]
        spectrum = spectrum_tensor.numpy()
        
        # Detect chemical peak locations using localized mean intensity values as prominence floor
        peaks, _ = find_peaks(spectrum, prominence=np.mean(spectrum))
        
        if len(peaks) > 0:
            # Evaluate peak envelope widths at 90% relative depth to capture complete peak bases
            widths = peak_widths(spectrum, peaks, rel_height=0.9)[0]
            if len(widths) > 0:
                current_max = np.max(widths)
                if current_max > max_width:
                    max_width = current_max
                    
    # Ceiling conversion to transform continuous width steps into integer counts
    suggested_kernel = int(np.ceil(max_width))
    
    # Enforce standard convolution symmetry constraints (odd kernel sizing)
    if suggested_kernel % 2 == 0:
        suggested_kernel += 1
        
    return max(3, suggested_kernel)