# import pytest
# import numpy as np
# import torch
# # These names are imported from your actual registry definitions
# from ims_contrastive_model.utils.Binners import BINNER_REGISTRY, IMSPyTorchBinner_Registry, IMSPyTorchInverseBinner
# def sample_mz_grid():
#     """Linear grid from 100 to 200 with 1.0 m/z steps."""
#     return np.linspace(100.0, 200.0, 101)

# def test_linear_binning_window_logic(sample_mz_grid):
#     """
#     Test: Checks if LinearBinning correctly assigns peaks to the nearest bin.
#     Checks: Window width and edge cases.
#     """
#     binner_cls = BINNER_REGISTRY["LinearBinning"]
#     binner = binner_cls(grid=sample_mz_grid)
    
#     # 1. Test precise bin hit
#     mz = np.array([105.0])
#     intensity = np.array([100.0])
#     vector = binner(mz, intensity)
#     assert vector[5] == 100.0 # Index 5 is exactly 105.0
    
#     # 2. Test rounding logic (0.4 shift)
#     mz_shift = np.array([105.4]) # Should still hit index 5 (105.0)
#     vector_shift = binner(mz_shift, intensity)
#     assert vector_shift[5] == 100.0
    
#     # 3. Test multiple peaks in one bin (Summation logic)
#     mz_multi = np.array([110.1, 110.2])
#     int_multi = np.array([50.0, 50.0])
#     vector_multi = binner(mz_multi, int_multi)
#     assert vector_multi[10] == 100.0 # Sum of intensities at 110.0

# def test_inverse_binning_registry_output():
#     """
#     Test: Iterates through Inverse Binner registry.
#     Ensures they return (mz, intensity) tuples and handle empty/sparse input.
#     """
#     grid = np.linspace(100.0, 200.0, 101)
    
#     # Sample decoded output (batch of 1 spectrum)
#     dummy_decoded = np.zeros(101)
#     dummy_decoded[10] = 50.0  # Peak at 110.0 m/z
#     dummy_decoded[20] = 100.0 # Peak at 120.0 m/z
    
#     for name in IMSPyTorchInverseBinner:
#         inv_binner_cls = BINNER_REGISTRY[name]
#         inv_binner = inv_binner_cls(grid=grid)
        
#         # Run inverse binning
#         mzs, ints = inv_binner(dummy_decoded)
        
#         # Assertions
#         assert isinstance(mzs, np.ndarray), f"{name} must return numpy mz array"
#         assert len(mzs) > 0, f"{name} failed to extract peaks"
#         assert 110.0 in mzs, f"{name} missed expected peak at 110.0"
        
#         if name == "TopPeaksInverseBinner":
#             # Test if it actually limits to Top N if requested
#             inv_binner.n_peaks = 1
#             mzs_top, _ = inv_binner(dummy_decoded)
#             assert len(mzs_top) == 1
#             assert mzs_top[0] == 120.0 # Highest peak

# def test_binner_empty_input(sample_mz_grid):
#     """
#     Test: Ensures binners do not crash on zero-intensity spectra.
#     """
#     binner = BINNER_REGISTRY["LinearBinning"](grid=sample_mz_grid)
#     vector = binner(np.array([]), np.array([]))
#     assert np.all(vector == 0)
#     assert len(vector) == len(sample_mz_grid)