"""
Technical Documentation for IMSContrastiveModel
Reference: "Efficient Compression of Mass Spectrometry Images via Contrastive Learning-Based Encoding"
Link: https://pubmed.ncbi.nlm.nih.gov/40689435/
"""

IMSContrastiveModel_DOC = """
    Main class for Contrastive Mass Spectrometry Imaging (MSI) compression and encoding.
    
    This model utilizes a Contrastive Autoencoder (CAE) to map high-dimensional MSI 
    spectra into a compact latent space. It leverages self-supervised learning 
    (InfoNCE loss) combined with reconstruction objectives to preserve both 
    global spatial structures and local chemical variances.

    Architecture Overview:
        1. Encoder: 1D-CNN layers reducing spectral bins to a latent manifold.
        2. Projector: Non-linear MLP mapping latent features to a hypersphere for contrastive task.
        3. Decoder: Transposed 1D-CNN reconstructing the original m/z signal.

    Loss Function Components:
        L_total = λ1*L_contrastive + λ2*L_mse + λ3*L_var + λ4*L_mean

        - Contrastive (InfoNCE): Maximizes agreement between augmented views of the same pixel.
        - Reconstruction (MSE): Ensures the latent space retains enough information to recover spectra.
        - Variance Regularization (L_var): Forces the standard deviation of latent features across 
          the batch to approach 1, preventing dimensional collapse.
        - Mean Regularization (L_mean): Penalizes the mean of latent activations to center 
          the distribution around zero.

    Example Hyperparameters:
        >>> hyperparameters = {
        ...    'channels': [1, 16, 32, 64, 128],  # Filters per layer
        ...    'kernels': [7, 5, 5, 3, 3],        # Receptive field per layer
        ...    'strides': [2, 2, 2, 2, 2]         # Downsampling factor
        ... }
    """

IMSContrastiveModel_init_DOC = """
        Initializes the model with MSI data provider and training parameters.

        Args:
            IMSLoader (IMSPyTorchDataset): CusDOCStom dataset handling .imzML resampling and binning.
            latent_dim (int): Dimensionality of the compressed manifold (e.g., 32, 64).
            epochs (int): Maximum number of training iterations.
            batch_size (int): Number of spectra per optimization step.
            lr (float): Learning rate for Adam optimizer.
            patience_limit (int): Early stopping patience.
            hyperparameters (dict): Manual CNN configuration. If None, suggest_cnn_configuration is used.
        """

IMSContrastiveModel_fit_DOC = """
        Executes the training pipeline.

        Builds the architecture based on the m/z grid size, initializes the dual-objective 
        loss, and runs the optimization loop with early stopping and learning rate scheduling.

        Args:
            save_dir (str): Path to store model weights (model_weights.pt) and config.json.
            
        Example:
            >>> model.fit(save_dir="./models/bladder_test")
        """

IMSContrastiveModel_transform_DOC = """
        Encodes the entire MSI image into the latent space.

        Iterates through the full IMSLoader without augmentation to produce a 
        static compressed representation of the tissue.

        Returns:
            np.ndarray: Matrix of shape [N_pixels, latent_dim].
        
        Example:
            >>> latent_space = model.transform()
            >>> print(latent_space.shape) # (17417, 64)
        """

IMSContrastiveModel_encode_DOC = """
        Maps a batch of raw spectra to the L2-normalized latent space.

        Args:
            x (torch.Tensor): Input spectra tensor of shape [Batch, Bins].

        Returns:
            np.ndarray: Latent embeddings.
        """

IMSContrastiveModel_decode_DOC = """
        Reconstructs spectral profiles from latent vectors.

        Args:
            z (torch.Tensor): Latent vectors of shape [Batch, latent_dim].

        Returns:
            np.ndarray: Reconstructed intensities [Batch, Bins].
        """
