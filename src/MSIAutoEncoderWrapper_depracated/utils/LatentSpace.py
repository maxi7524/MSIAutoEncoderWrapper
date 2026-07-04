import numpy as np

def build_latent_grid(embeddings: np.ndarray, coordinates: np.ndarray):
    """
    Constructs a structured latent space image from flat embeddings and 0-indexed coordinates.
    
    Args:
        embeddings: Latent vectors of shape (N, Latent_Dim).
        coordinates: Array of indices (N, 3) representing [x, y, z].
        
    Returns:
        np.ndarray: Grid of shape (X, Y, Z, C).
    """
    # Determine grid boundaries from 0-indexed coordinates
    x_max = int(coordinates[:, 0].max()) + 1
    y_max = int(coordinates[:, 1].max()) + 1
    z_max = int(coordinates[:, 2].max()) + 1
    latent_dim = embeddings.shape[1]

    # Initialize (X, Y, Z, C) grid
    grid = np.zeros((x_max, y_max, z_max, latent_dim), dtype=np.float32)

    # Map each vector to its spatial position
    for i in range(len(embeddings)):
        x, y, z = coordinates[i].astype(int)
        grid[x, y, z, :] = embeddings[i]
        
    return grid