import pytest
import numpy as np
from ims_contrastive_model.utils.LatentSpace import build_latent_grid

def test_latent_grid_spatial_reconstruction():
    """
    Test: Verifies that build_latent_grid reconstructs a full 2D grid from sparse coordinates.
    Scenario: A typical MSI slice of 260x134 pixels.
    """
    # 1. Define dimensions and latent space size
    NX, NY, NZ, C = 260, 134, 1, 64
    
    # 2. Simulate sparse data: Only top-left, bottom-right, and a middle point
    # We use 0-indexed coordinates as per model convention
    coordinates = np.array([
        [0, 0, 0],          # Top-left
        [259, 133, 0],      # Bottom-right
        [100, 50, 0]        # Random point
    ])
    
    # Random embeddings for these 3 points
    embeddings = np.random.rand(3, C).astype(np.float32)
    
    # 3. Build the grid
    grid = build_latent_grid(embeddings, coordinates)
    
    # 4. VERIFY SHAPE: Must be derived from max coordinates
    assert grid.shape == (NX, NY, NZ, C), f"Grid shape {grid.shape} mismatch with expected (260, 134, 1, 64)"
    
    # 5. VERIFY CONTENT: Check if data is in the right spot
    assert np.allclose(grid[0, 0, 0, :], embeddings[0]), "Top-left pixel mismatch"
    assert np.allclose(grid[259, 133, 0, :], embeddings[1]), "Bottom-right pixel mismatch"
    assert np.allclose(grid[100, 50, 0, :], embeddings[2]), "Center pixel mismatch"
    
    # 6. VERIFY VACUUM: Unmapped pixels must be zero
    assert np.sum(grid[1, 1, 0, :]) == 0, "Empty pixel should contain only zeros"

def test_latent_grid_3d_support():
    """
    Test: Ensures the grid builder handles multiple Z-slices (Volumes).
    """
    embeddings = np.random.rand(2, 16).astype(np.float32)
    coords = np.array([[0, 0, 0], [0, 0, 1]]) # Two slices
    
    grid = build_latent_grid(embeddings, coords)
    
    assert grid.shape == (1, 1, 2, 16), "Failed to recognize Z-dimension depth"
    assert np.allclose(grid[0, 0, 1, :], embeddings[1])