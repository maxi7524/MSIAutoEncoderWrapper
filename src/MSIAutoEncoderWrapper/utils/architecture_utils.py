import numpy as np
from scipy.signal import find_peaks, peak_widths


# TODO - fasten the function
def estimate_max_peak_width(MSILoader, sample_size=100):
    """
    Analyzes a sample of spectra to find the largest peak envelope width (in bins).
    This helps in selecting the size of the first kernel in the CNN.[cite: 8]
    """
    total_spectra = len(MSILoader)
    # Randomly select indices of spectra for analysis (for performance efficiency)[cite: 5]
    indices = np.random.choice(total_spectra, min(sample_size, total_spectra), replace=False)
    
    max_width = 0
    
    for idx in indices:
        # Retrieve the normalized spectrum from the Loader
        _, spectrum = MSILoader[idx]
        spectrum = spectrum.numpy()
        
        # Find peaks 
        peaks, properties = find_peaks(spectrum, prominence=np.mean(spectrum))
        
        if len(peaks) > 0:
            # Calculate peak widths at 10% of their height (envelope at the base)
            widths = peak_widths(spectrum, peaks, rel_height=0.9)[0]
            if len(widths) > 0:
                current_max = np.max(widths)
                if current_max > max_width:
                    max_width = current_max
                    
    # Return as int, minimum of 3 (to ensure the kernel is meaningful)
    # Round up to the nearest odd number for kernel symmetry
    suggested_kernel = int(np.ceil(max_width))
    if suggested_kernel % 2 == 0:
        suggested_kernel += 1
        
    return max(3, suggested_kernel)