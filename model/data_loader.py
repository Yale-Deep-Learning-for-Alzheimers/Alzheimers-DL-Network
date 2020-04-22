import os
import sys
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import nibabel as nib


class MRIData(Dataset):
    """
    MRI data
    """

    def __init__(self, root_dir, data_file):
        """
        Args:
            root_dir (string): directory of all the images
            NOTE: Since we have key-[value]* pairs, we'll probably need to modify this to the root of the dictionary
            data_file (string): file name of the .csv
        """
        self.root_dir = root_dir
        self.data_file = data_file

    def __len__(self):
        """
        Required by DataLoader
        """
        return sum(1 for line in open(self.data_file)) # the number of entries in the .csv
        """ TODO: are all the subjects in the CSVs' rows unique? I think it looked like it """

    def __getitem__(self, index):
        """
        Allows indexing of dataset; required by DataLoader
        """
        pass
    """ TODO: use nib to get the file and load it; should we convert to tensor here and return the tensor? """


class DataLoader(object):
    """
    Handles data
    """

    def __init__(self):
        pass
