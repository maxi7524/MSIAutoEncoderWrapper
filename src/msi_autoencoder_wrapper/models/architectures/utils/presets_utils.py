"""
Statistical utility algorithms executing analytical computations over structural dataset matrices.
"""


import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing import Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ....core.mixins.active_context.active_context_mixin import ActiveContextProxy
from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error

# Logger initialization
logger = get_custom_logger(__name__)


def estimate_max_peak_width(
    active_context: Any,
    sample_size: int = 10000,
    random_seed: Optional[int] = None,
) -> int:
    """
    Estimates the maximum peak envelope width across a random sample of pixel profiles.

    Detects significant peaks and measures their envelope widths at 90 percent
    relative height, matching the original data-driven preset heuristic.

    :param active_context: Context exposing an image reader and forward binner.
    :type active_context: Any
    :param sample_size: Maximum number of spectra inspected.
    :type sample_size: int
    :param random_seed: Optional reproducible sampling seed.
    :type random_seed: Optional[int]
    :return: Largest observed envelope width rounded up to an odd kernel size.
    :rtype: int
    :raises ValidationError: If sampling or required context components are invalid.
    """
    if sample_size < 1:
        raise_validation_error(
            context_name="PeakWidthEstimator",
            message="sample_size must be at least one.",
        )
    reader = getattr(active_context, "reader", None)
    binner = getattr(active_context, "binner", None)
    if reader is None or binner is None:
        raise_validation_error(
            context_name="PeakWidthEstimator",
            message="An active image reader and binner are required.",
        )

    total_pixels = reader.GetNumberOfSpectra()
    if total_pixels < 1:
        raise_validation_error(
            context_name="PeakWidthEstimator",
            message="The active image does not contain spectra.",
        )
    actual_sample_size = min(total_pixels, sample_size)
    
    logger.debug("Initiating statistical peak width evaluation over a pool of %s pixels.", actual_sample_size)
    
    # Execution scanning loop
    ## Select pseudorandom pixel indices across the total physical layout boundaries
    random_generator = np.random.default_rng(random_seed)
    indices = random_generator.choice(
        total_pixels,
        size=actual_sample_size,
        replace=False,
    )
    max_detected_width = 0.0
    
    for idx in indices:
        xs, ys = reader.GetSpectrum(int(idx))
        spectrum_arr = binner(xs=xs, ys=ys)
        peaks, _ = find_peaks(
            spectrum_arr,
            prominence=max(float(np.mean(spectrum_arr)), 0.0),
        )
        if peaks.size == 0:
            continue
        widths = peak_widths(spectrum_arr, peaks, rel_height=0.9)[0]
        if widths.size > 0:
            max_detected_width = max(max_detected_width, float(np.max(widths)))

    final_kernel_suggestion = max(3, int(np.ceil(max_detected_width)))
    if final_kernel_suggestion % 2 == 0:
        final_kernel_suggestion += 1
    logger.info("Statistical reflection completed. Suggested baseline kernel width: %s bins", final_kernel_suggestion)
    
    return final_kernel_suggestion
