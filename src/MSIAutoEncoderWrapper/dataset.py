"""
ims_contrastive_model/dataloader.py
----------------------------------
Contains code for dataloader for m2aia ims_contrastive_model
"""

# base
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
# MSI
import m2aia as m2
# binning module
from .utils.Binners import MSIPyTorchBinner


# TODO - we could use certain *mask* (so we do not batch noise pixel ~ match faster learning)


class MSIPyTorchDataset(Dataset):
    """
    An adapter class that bridges the M2AIA ImzMLReader with the PyTorch Dataset ecosystem.

    This class is responsible for loading raw Mass Spectrometry Imaging (MSI) data 
    and managing the local training samples. It acts as a wrapper around the 
    m2aia reader to ensure data is served in a format compatible with Deep Learning.

    **Why use Binning?**
    Raw MSI data often has irregular m/z axes (different m/z values per pixel). 
    Convolutional Neural Networks (CNNs) rely on spatial consistency—meaning a 
    specific index in the input tensor must always represent the same m/z value. 
    This class utilizes binners from ``utils.Binners`` to project irregular 
    spectra onto a fixed, common grid, allowing CNN kernels to learn 
    local chemical patterns effectively.

    **Data Handling & Safety:**
    - **Spatial Identification**: The dataset returns the ``spatial_idx`` along with 
      the spectral tensor, enabling the reconstruction of the latent space back 
      into the original X,Y tissue coordinates.
    - **Error Resilience**: If a spectrum at a specific index is corrupted or 
      missing in the imzML file, the class returns a zero-filled tensor of the 
      correct dimensions. This prevents training crashes and maintains batch consistency.

    :param m2aia_img: Active m2aia reader object.
    :type m2aia_img: m2aia.ImzMLReader
    :param Binner: The binner object used to standardize the m/z axis.
    :type Binner: MSIPyTorchBinner
    """
    def __init__(self, 
                # obligatory 
                m2aia_img: m2.ImzMLReader,
                Binner: MSIPyTorchBinner
            ):
        """
        Initializes the MSI dataset with a fixed m/z grid.

        Args:
            m2aia_img (m2.ImzMLReader): An opened and loaded M2AIA reader object.
            resampling_method (str): Statistical method for binning ('mean', 'max', 'sum').
                Defaults to 'mean'.
            mz_min (float, optional): Lower bound of the m/z grid. 
                If None, extracted from metadata.
            mz_max (float, optional): Upper bound of the m/z grid. 
                If None, extracted from metadata.
            mz_resolution (float, optional): The step size between m/z bins. 
                If None, extracted from 'pixel size z' metadata.

        Warning:
            The 'mz_resolution' is critical for memory and performance. Too small 
            resolution results in very large tensors, potentially causing OOM errors.
        """
        ## image
        self._img = m2aia_img
        self._Binner = Binner
        # self.data_path = Path(data_path)
    
    # ---------------------
    # dataset essentials 
    # ---------------------

    def __len__(self):
        """Returns the total number of spectra available in the m2aia object."""
        return self.img.GetNumberOfSpectra()

    def __getitem__(self, spatial_idx):
        """
        Retrieves, bins, and converts a spectrum to a standardized PyTorch tensor.

        :param spatial_idx: The 0-based index of the spectrum in the m2aia object.
        :returns: A tuple of (spatial_index, spectral_tensor).
        :rtype: tuple(int, torch.Tensor)
        """
        # load data
        try: 
            ## Sample is single spectrum 
            xs, ys = self.img.GetSpectrum(spatial_idx)
            ## map to common axis
            mapped_val = self.Binner(xs=xs, ys=ys)
            return spatial_idx, torch.tensor(mapped_val, dtype=torch.float32)
        except:
            ## if GetSpectrum(idx) is None
            return spatial_idx, torch.zeros(len(self.GetGridXAxis), dtype=torch.float32)
    
    # ---------------------
    # helpers
    # ---------------------


    # ---------------------
    # getters and setters (functions for m2aia convenience)
    # ---------------------
    
    @property
    def img(self):
        return self._img

    @property
    def Binner(self):
        return self._Binner

    # --- image part ---

    def GetXMin(self):
        return self.img.GetXAxis[0]
    
    def GetXMax(self):
        return self.img.GetXAxis[-1]
    
    def GetXAxis(self):
        return self.img.GetXAxis()
    
    def GetXAxisDepth(self):
        return self.img.GetXAxisDepth()

    # --- binner part ---

    def GetGridXMin(self):
        return self.Binner.GetXMin()

    def GetGridXMax(self):
        return self.Binner.GetXMax()
    
    def GetGridXAxis(self):
        return self.Binner.GetXAxis()
    
    def GetGridXAxisDepth(self):
        return self.Binner.GetXAxisDepth()
    
    # @property
    # def mz_resolution(self):
    #     return self._mz_resolution

    # @property
    #TODO - to powinno zwracać nazwę tej używanej metody 
    # def resampling_method(self):
    #     return self._resampling_method
    
    


    





        

    
    
