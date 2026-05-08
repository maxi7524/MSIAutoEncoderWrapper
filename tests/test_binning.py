import pytest
import numpy as np
from MSIAutoEncoderWrapper.utils.Binners import (
    MSIPyTorchBinnerRegistry, 
    MSIPyTorchInverseBinnerRegistry,
    LinearBinning
)

# Configuration for forward binners
FORWARD_BINNER_PARAMS = {
    "LinearBinning": {
        "xs_min": 100,
        "xs_max": 200,
        "xs_res": 1.0,
        "ys_agg_method": "mean"
    }
}

# Configuration for inverse binners
INVERSE_BINNER_PARAMS = {
    "NotEmptyInverseBinner": {},
    "TopPeaksInverseBinner": {
        "max_bins": 5,
        "window_size": 2,
        "threshold": 0.1
    }
}

@pytest.fixture
def shared_linear_binner():
    """Provides a standard initialized LinearBinning object for inverse tests."""
    return LinearBinning(xs_min=100, xs_max=200, xs_res=1.0)

@pytest.mark.parametrize("binner_name", MSIPyTorchInverseBinnerRegistry.keys())
class TestForwardBinners:
    """
    Suite for testing forward binning classes that map raw spectra to a grid.
    """

    def test_forward_binning_logic(self, binner_name):
        """
        Test: Mapping raw (xs, ys) to a fixed grid.
        Expectation: 
            - Should return a binned intensity array of predictable length.
            - NaNs should be converted to 0.0.
        Why: Ensures CNNs receive spatially consistent data depth regardless of raw input size.
        """
        binner_class = MSIPyTorchInverseBinnerRegistry[binner_name]
        params = FORWARD_BINNER_PARAMS[binner_name]
        binner = binner_class(**params)
        
        # Create raw data: two points falling into the same bin (around 105.0)
        xs = np.array([105.1, 105.4, 300.0]) # 300 is out of range
        ys = np.array([10.0, 20.0, 50.0])
        
        binned_ys = binner(xs, ys)
        
        # Grid from 100 to 200 with res 1.0 has 101 points
        assert len(binned_ys) == binner.GetXAxisDepth()
        assert binned_ys[6] == 15.0  # mean of 10 and 20 at m/z 105 (index 5)
        assert binned_ys[-1] == 0.0  # out of range value should not be there

    def test_forward_getters(self, binner_name):
        """
        Test: Basic grid metadata retrieval.
        Expectation: GetXMin, GetXMax, and GetXAxis must return values consistent with initialization.
        Why: Architecture suggests (SetHyperparameters) depend on correct m/z axis reporting.
        """
        binner_class = MSIPyTorchInverseBinnerRegistry[binner_name]
        params = FORWARD_BINNER_PARAMS[binner_name]
        binner = binner_class(**params)
        
        assert binner.GetXMin() == params["xs_min"]
        assert binner.GetXMax() == params["xs_max"]
        assert len(binner.GetXAxis()) == binner.GetXAxisDepth()

    def test_forward_config(self, binner_name):
        """
        Test: GetConfig() dictionary content.
        Expectation: Dictionary must contain all keys used in __init__.
        Why: Critical for model saving/loading and ensuring identical binning parameters.
        """
        binner_class = MSIPyTorchInverseBinnerRegistry[binner_name]
        params = FORWARD_BINNER_PARAMS[binner_name]
        binner = binner_class(**params)
        
        config = binner.GetConfig()
        for key in params:
            assert config[key] == params[key], f"Missing or wrong config value for {key}"


@pytest.mark.parametrize("inv_name", MSIPyTorchBinnerRegistry.keys())
class TestInverseBinners:
    """
    Suite for testing inverse binning classes that reduce grid data to sparse peaks.
    """

    def test_inverse_binning_logic(self, inv_name, shared_linear_binner):
        """
        Test: Reducing dense grid_ys to sparse (xs, ys) representation.
        Expectation: 
            - Resulting arrays must have equal length.
            - Should respect thresholds and limits (like max_bins).
        Why: Used for visualization and saving sparse MSI files to reduce storage size.
        """
        inv_class = MSIPyTorchBinnerRegistry[inv_name]
        params = INVERSE_BINNER_PARAMS[inv_name]
        inv_binner = inv_class(Binner=shared_linear_binner, **params)
        
        # Create grid with 3 peaks
        grid_ys = np.zeros(shared_linear_binner.GetXAxisDepth())
        grid_ys[10] = 5.0
        grid_ys[20] = 10.0
        grid_ys[50] = 1.0
        
        red_xs, red_ys = inv_binner(grid_ys)
        
        assert len(red_xs) == len(red_ys)
        assert len(red_xs) <= shared_linear_binner.GetXAxisDepth()
        
        if inv_name == "TopPeaksInverseBinner":
            # With window_size=2, each peak is 5 points (idx-2 to idx+2)
            # Max bins = 5, so it should keep only the top peak (idx 20)
            assert len(red_ys) == 5
            assert np.max(red_ys) == 10.0

    def test_inverse_config(self, inv_name, shared_linear_binner):
        """
        Test: GetConfig() for inverse binners.
        Expectation: Must return a dictionary containing the parent Binner's configuration.
        Why: Ensures that the inverse transformation is always tied to the correct m/z grid definition.
        """
        inv_class = MSIPyTorchBinnerRegistry[inv_name]
        params = INVERSE_BINNER_PARAMS[inv_name]
        inv_binner = inv_class(Binner=shared_linear_binner, **params)
        
        config = inv_binner.GetConfig()
        assert isinstance(config, dict)
        assert "Binner" in config
        assert config["Binner"]["xs_min"] == shared_linear_binner.GetXMin()