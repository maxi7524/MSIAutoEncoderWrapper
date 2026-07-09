"""
Statistical utility algorithms executing analytical computations over structural dataset matrices.
"""

import numpy as np
from typing import Any
from ....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


def estimate_max_peak_width(msi_dataset: Any, sample_size: int = 10000) -> int:
    """
    Estimates the maximum peak envelope width across a random sample of pixel profiles.

    Analyzes consecutive bins thresholds to gauge spatial dimensions parameters.

    :param msi_dataset: Instantiated concrete dataset complying with the MSIBaseDataset contract.
    :type msi_dataset: Any
    :param sample_size: Upper limit boundary defining maximum sampled pixel profiles, defaults to 10000.
    :type sample_size: int
    :return: Estimated peak width expressed as absolute mass-to-charge grid channels count.
    :rtype: int
    """
    total_pixels = len(msi_dataset)
    actual_sample_size = min(total_pixels, sample_size)
    
    logger.debug("Initiating statistical peak width evaluation over a pool of %s pixels.", actual_sample_size)
    
    # Execution scanning loop
    ## Select pseudorandom pixel indices across the total physical layout boundaries
    indices = np.random.choice(total_pixels, size=actual_sample_size, replace=False)
    max_detected_width = 5
    
    for idx in indices:
        try:
            ### Unpack standard dataset vector entries
            _, spectrum_tensor = msi_dataset[idx]
            spectrum_arr = spectrum_tensor.numpy()
            
            ### Compute continuous active signal regions mapping sequences
            active_mask = spectrum_arr > (np.mean(spectrum_arr) * 1.5)
            consecutive_blocks = np.diff(np.where(np.concatenate(([0], active_mask, [0])))[0])[::2]
            
            if consecutive_blocks.size > 0:
                current_max = int(np.max(consecutive_blocks))
                if current_max > max_detected_width:
                    max_detected_width = current_max
        except Exception:
            continue
            
    # Cap boundaries thresholds to prevent extreme dimension values anomalies
    final_kernel_suggestion = min(max(max_detected_width, 9), 35)
    logger.info("Statistical reflection completed. Suggested baseline kernel width: %s bins", final_kernel_suggestion)
    
    return final_kernel_suggestion