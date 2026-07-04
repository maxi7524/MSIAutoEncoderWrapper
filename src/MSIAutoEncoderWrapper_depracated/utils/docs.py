

MSIAutoEncoder_main_DOC = """
        High-level Manager for the Contrastive MSI Autoencoder lifecycle.

        This class orchestrates the interaction between the neural network architecture, 
        the loss criteria, and the data loaders. It is designed to guide the user 
        through two primary workflows:

        The model supports advanced operations like full-image transformation, 
        latent space grid generation, and compressed I/O (.npz to reconstructed .imzML).

        **Create model**

                **Workflow A: Creating and Training a New Model**
                1. Initialize the class with a path.
                2. Set the Dataset, Architecture, and Criterion.
                3. Run ``.fit()`` to train and ``.save()`` to store the result.

                **Workflow B: Loading an Existing Model**
                1. Initialize the class with the path to the model folder.
                2. Run ``.load()`` to automatically reconstruct the Architecture, Binners, 
                and weights from the stored JSON configuration.

        **NEXT OPERATIONS**
        TODO


    """

MSIAutoEncoder_fit_DOC = """
        Runs the full training pipeline using the ``train_model`` engine.

        This method prepares the DataLoader, handles hardware allocation, 
        and sets up persistence callbacks.




        :param save_dir: Directory to save weights and logs. Defaults to model path.
        :type save_dir: str | Path, optional
        :param continue_training: If True, attempts to load 'model_latest.pt' 
                                  to resume training from the last state.
        :type continue_training: bool
        :param train_model_config: Overrides for training params (epochs, lr, etc.).
        :type train_model_config: dict, optional
        :param TorchLoader_config: Overrides for DataLoader (batch_size, num_workers).
        :type TorchLoader_config: dict, optional
        """

MSIAutoEncoder_transform_DOC = """
        Processes the entire dataset through the encoder to create a latent map.

        :returns: Matrix of latent embeddings for all pixels. Shape: ``[N_pixels, latent_dim]``.
        :rtype: np.ndarray
        """

MSIAutoEncoder_encode_DOC = """
        Encodes raw spectra into the latent space.

        :param x: Input spectra. Can be a single spectrum or a batch.
                  Expected shape: ``[Batch, Bins]`` or ``[Bins]``.
        :type x: torch.Tensor | np.ndarray
        :returns: L2-normalized latent embeddings of shape ``[Batch, latent_dim]``.
        :rtype: np.ndarray
        """

MSIAutoEncoder_decode_DOC = """
        Decodes latent vectors back into the spectral domain.

        :param z: Latent vectors of shape ``[Batch, latent_dim]``.
        :type z: torch.Tensor | np.ndarray
        :param grid_xs: If True, returns raw intensities on the internal grid. 
                        If False, uses ``InverseBinner`` to return (m/z, intensities).
        :type grid_xs: bool
        :returns: Reconstructed spectra.
        :rtype: np.ndarray | tuple(np.ndarray, np.ndarray)
        """
