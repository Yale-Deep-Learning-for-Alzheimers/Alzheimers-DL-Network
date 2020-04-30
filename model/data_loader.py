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
        Returns length of dataset       (required by DataLoader)
        """
        return len(self.data_array) # the number of patients in the dataset

    def __getitem__(self, index):
        """
        Allows indexing of dataset      (required by DataLoader)
        Returns a tensor that contains the patient's MRI neuroimages and their diagnoses (AD or MCI)
        """
        
        # Get current_patient, where [0] is their ID and [1] is their list of images
        # NOTE: you can't index current_patient[1] to get the list of images, you just index current_patient itself... not sure why this works but it works
        current_patient = self.data_array[index]
        # List to store the individual image tensors
        images_list = []
        # The last element in the current patient's array is the classification
        patient_label = current_patient.pop()
        # For each image path, process the .nii image using nibabel
        for image_path in current_patient:
            print(image_path)
            file_name = os.path.join(self.root_dir, image_path)
            neuroimage = nib.load(file_name)
            # Extract the N-D array containing the image data from the nibabel image object
            image_data = neuroimage.get_fdata()
            image_data_tensor = torch.Tensor(image_data) # Convert image data to a tensor
            images_list.append(image_data_tensor)

        # Convert the list of individual image tensors to a tensor itself
        images_tensor = torch.stack(images_list)

        # Return a dictionary with the images tensor and the label
        image_dict = {'images': images_tensor, 'label': patient_label}
        # NOTE: alternative approach is to just have the classification be the first element in the images_tensor

        return image_dict
        