""" Pytorch implementation of a CNN LSTM, modified from a pytorch tutorial
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# For reproducibility for testing purposes. Delete during actual training.
# torch.manual_seed(1) 

class Network(nn.Module):
    """ CNN LSTM to classify ADNI data. Specify:
        + embedding dimension, the number of channels each input image has (likely 1).
        + input_size, the shape of the input in a tuple: (Depth, Height, Width)
        + output_size, a scalar (like 4) that specifies the number of predictions the network should make."""

    def __init__(self, input_channels, input_shape, output_size, lstm_layers=1):
        super(Network, self).__init__()

        print("Initializing hyperparameters...")

        def dimensions_after_convolution(kernel, stride, padding, input_shape):
            # helper function for automatic calculation of hyperparameters
            output_depth = math.floor((input_shape[0] + 2 * padding - kernel + 1) / stride + (
                    stride - 1) / stride)
            output_height = math.floor((input_shape[1] + 2 * padding - kernel + 1) / stride + (
                    stride - 1) / stride)  # the total height is the input plus twice the padding
            output_width = math.floor((input_shape[2] + 2 * padding - kernel + 1) / stride + (stride - 1) / stride)
            return output_depth, output_height, output_width

        # CNN for feature selection and encoding
        # CNN-specific hyperparameters
        kernel_size = 4
        padding = 0

        # The input and output
        self.convolution1 = nn.Conv3d(input_channels, 10,kernel_size,padding=padding)
        current_shape = dimensions_after_convolution(kernel_size,1,padding,input_shape)
        self.pool1 = nn.MaxPool3d(kernel_size)
        current_shape = dimensions_after_convolution(kernel_size,kernel_size,padding,current_shape)
        self.convolution2 = nn.Conv3d(10, 5,kernel_size,padding=padding)
        current_shape = dimensions_after_convolution(kernel_size, 1, padding, current_shape)
        self.pool2 = nn.MaxPool3d(kernel_size)
        current_shape = dimensions_after_convolution(kernel_size, kernel_size, padding, current_shape)
        self.convolution3 = nn.Conv3d(5, 1, kernel_size, padding=padding)
        current_shape = dimensions_after_convolution(kernel_size, 1, padding, current_shape)

        # LSTM to combine feature encoding from above with feature encodings from past networks
        # The input dimension is the volume of the remaining 3d image after convolution and pooling.
        lstm_input_dimensions = current_shape[0]*current_shape[1]*current_shape[2]
        print(f"For the specified shape, the LSTM input dimension has been calculated at {lstm_input_dimensions}.")
        if lstm_input_dimensions>1000: print("This seems very large. Perhaps you should add some more layers to your network, or increase the kernel size.")
        self.lstm = nn.LSTM(lstm_input_dimensions, lstm_input_dimensions,lstm_layers)
        # The linear layer that maps from hidden state space to prediction space
        self.prediction_converter = nn.Linear(lstm_input_dimensions, output_size)
        self.num_layers = lstm_layers
        self.hidden_dimensions = lstm_input_dimensions

    def init_hidden(self,batch_size=1):
        # Used for initializing LSTM weights between patients.
        return (torch.zeros(self.num_layers,batch_size, self.hidden_dimensions),
                torch.zeros(self.num_layers,batch_size, self.hidden_dimensions))

    def forward(self, MRI):
        feature_space = self.convolution3(self.pool2(self.convolution2(self.pool1(self.convolution1(MRI)))))
        # Flatten the output layers from the CNN into a 1d tensor
        lstm_in = torch.cat([torch.flatten(image[0])[...,None] for image in feature_space],axis=0).view(feature_space.shape[0],1,-1) # This assumes one output channel from CNN
        lstm_out, self.hidden = self.lstm(lstm_in) # assuming mini-batch of 1
        # To feed the final LSTM layer through the last layer, we need to convert the multidimensional output to
        # a single dimensional tensor.
        dense_conversion = self.prediction_converter(lstm_out)
        dense_conversion = torch.squeeze(dense_conversion)

        return dense_conversion

# Testing. Run random data through network to ensure that everything checks out.
if __name__ == "__main__":
    big_net = Network(1,(100,100,100),3)
    iou = torch.randn(4,1,100,100,100)
    preds = big_net(iou)
    print(preds)