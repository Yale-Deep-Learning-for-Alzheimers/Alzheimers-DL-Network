import os
import sys # possibly don't need
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import nibabel as nib


class MRIData(Dataset):
    """
    MRI data
    The dictionaries AD_Img_Dict.pkl and MCI_Img_Dict.pkl contain key-value
      pairs of the following:
      key:      subject ID
      value:    paths to relevant images (i.e. multiple images per subject)
    These dictionaries are converted to arrays and passed into this dataset,
    where the paths will be accessed and their neuroimages processed into tensors.
    """

    def __init__(self, root_dir, data_array):
        """
        Args:
            root_dir (string): directory of all the images
            data_array (list): array that contains one [key, [value]*] entry for each patient,
                               where key:       subject ID
                                     value:     paths to patient's MRI .nii neuroimages
        """
        self.root_dir = root_dir
        self.data_array = data_array

    def __len__(self):
        """
        Required by DataLoader
        """
        return len(self.data_array) # the number of patients in the dataset

    def __getitem__(self, index):
        """
        Allows indexing of dataset; required by DataLoader
        Returns a tensor that contains the patient's MRI neuroimages and their diagnoses (AD or MCI)
        """
        # Get current_patient, where [0] is their ID and [1] is their list of images
        current_patient = self.data_array[index]
        # TODO: create a tensor to store the individual image tensors
        # For each image path, process the .nii image using nibabel
        for image_path in current_patient[1]:
            file_name = os.path.join(self.root_dir, image_path)
            neuroimage = nib.load(file_name)
            # TODO: convert neuroimage to a tensor
            # TODO: add to main tensor
        
        pass


class DataLoader(object):
    """
    Handles data
    """

    def __init__(self):
        pass
