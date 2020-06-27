import os
import sys # possibly don't need
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import nibabel as nib
from scipy import ndimage

# Dimensions of neuroimages after resizing
STANDARD_DIM1 = 200
STANDARD_DIM2 = 200
STANDARD_DIM3 = 150

# Maximum number of images per patient
MAX_NUM_IMAGES = 10

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
        current_patient = self.data_array[index]
        # List to store the individual image tensors
        images_list = []
        # The last element in the current patient's array is the classification
        patient_label = current_patient.pop()
        print(patient_label)
        # For each image path, process the .nii image using nibabel
        for image_path in current_patient:
            print(image_path) #FIXME: delete this
            file_name = os.path.join(self.root_dir, image_path)
            neuroimage = nib.load(file_name) # Loads proxy image
            # Extract the N-D array containing the image data from the nibabel image object
            image_data = neuroimage.get_fdata() # Retrieves array data
            # Resize and interpolate image
            image_size = image_data.shape # Store dimensions of N-D array
            current_dim1 = image_size[0]
            current_dim2 = image_size[1]
            current_dim3 = image_size[2]
            # Calculate scale factor for each direction
            scale_factor1 = STANDARD_DIM1 / float(current_dim1)
            scale_factor2 = STANDARD_DIM2 / float(current_dim2)
            scale_factor3 = STANDARD_DIM3 / float(current_dim3)
            # Resize image (spline interpolation)
            image_data = ndimage.zoom(image_data, (scale_factor1, scale_factor2, scale_factor3))
            print("Resize success") #FIXME: delete this
            # Convert image data to a tensor
            image_data_tensor = torch.Tensor(image_data) 
            images_list.append(image_data_tensor)
        
        # Add padding to make all final tensors the same size
        num_images = len(images_list)
        while (len(images_list) < MAX_NUM_IMAGES):
            padding_array = np.zeros((STANDARD_DIM1, STANDARD_DIM2, STANDARD_DIM3))
            padding_tensor = torch.Tensor(padding_array)
            images_list.append(padding_tensor)

        if (len(images_list) > MAX_NUM_IMAGES):
            print("Error: More than 10 images for one individual patient. Update MAX_NUM_IMAGES in data_loader.py")

        # Convert the list of individual image tensors to a tensor itself
        images_tensor = torch.stack(images_list,dim=0)

        # Return a dictionary with the images tensor and the label
        image_dict = {'images': images_tensor, 'label': patient_label, 'num_images':num_images}

        return image_dict
