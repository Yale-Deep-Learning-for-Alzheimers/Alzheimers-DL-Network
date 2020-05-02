""" Pytorch implementation of a CNN LSTM, modified from a pytorch tutorial
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
# torch.manual_seed(1) # for reproducibility for testing purposes. Delete during actual training.

""" moved to evaluate.py
# ============= Hyperparameters ===================
batch_size = 64

# Dimensionality of the data outputted by the LSTM,
# forwarded to the final dense layer. THIS IS A GUESS CURRENTLY.
# TODO: Find a sensible output size for the LSTM, perhaps by reviewing literature.
LSTM_output_size = 16

input_size = 1000 # Size of the processed MRI scans fed into the CNN. TODO: Change

output_dimension = 1 # the number of predictions the model will make
learning_rate = 0.1
training_epochs = 50
# Other hyperparameters unlisted: the depth of the model, the kernel size, the padding, the channel restriction.


# ========== Import Training Data Here ==============
# expected format:
# training_data stores batches of MRI's and classifications like this: [batch,batch,batch] : )
# each batch should be in form
# [Bunch of MRIs, Bunch of Classifications]
# and each 'bunch' in the batch should be grouped by patient
# Bunch of MRIs = [Patient 1 MRIs, Patient 2 MRIs,...]
# Bunch of Classifications = [Patient 1 classifications, Patient 2 Classifications...]
# the Classifications should be binary 0,1 probabilities in output_dimension dimensions. Perhaps something like this:
# [chance_of_normality: 0 , chance of MCI: 0, chance of AD: 1]

training_data = ...

"""

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

        # CNN for feature selection and encoding.

        # CNN Specific Hyperparameters: TODO: Optimize! These are guesses.
        kernel_size = 4
        padding = 0



        # The input and output
        self.convolution1 = nn.Conv3d(input_channels, 200,kernel_size,padding=padding) #TODO: Optimize ALL of these when Data Size is known
        current_shape = dimensions_after_convolution(kernel_size,1,padding,input_shape)
        # We may want to increase the channel size from input to gain better performance.
        self.pool1 = nn.MaxPool3d(kernel_size)
        current_shape = dimensions_after_convolution(kernel_size,kernel_size,padding,current_shape)

        self.convolution2 = nn.Conv3d(200, 100,kernel_size,padding=padding)
        current_shape = dimensions_after_convolution(kernel_size, 1, padding, current_shape)

        self.pool2 = nn.MaxPool3d(kernel_size)
        current_shape = dimensions_after_convolution(kernel_size, kernel_size, padding, current_shape)

        self.convolution3 = nn.Conv3d(100, 1, kernel_size, padding=padding)
        current_shape = dimensions_after_convolution(kernel_size, 1, padding, current_shape)

        # LSTM to combine feature encoding from above with feature encodings from past networks
        # the input dimension is the volume of the remaining 3d image after convolution and pooling.
        lstm_input_dimensions = current_shape[0]*current_shape[1]*current_shape[2]
        print(f"For the specified shape, the LSTM input dimension has been calculated at {lstm_input_dimensions}.")
        if lstm_input_dimensions>1000: print("This seems very large. Perhaps you should add some more layers to your network, or increase the kernel size.")
        self.lstm = nn.LSTM(lstm_input_dimensions, lstm_input_dimensions,lstm_layers)

        # The linear layer that maps from hidden state space to prediction space
        self.prediction_converter = nn.Linear(lstm_input_dimensions, output_size)

        self.num_layers = lstm_layers
        self.hidden_dimensions = lstm_input_dimensions
        # self.hidden = self.init_hidden(3)

    def init_hidden(self,batch_size):
        # Used for initializing LSTM weights between patients.
        return (torch.zeros(self.num_layers,batch_size, self.hidden_dimensions),
                torch.zeros(self.num_layers,batch_size, self.hidden_dimensions))

    def forward(self, MRI):
        feature_space = self.convolution3(self.pool2(self.convolution2(self.pool1(self.convolution1(MRI)))))
        # flatten the output layers from the CNN into a 1d tensor
        print(feature_space.shape)
        lstm_in = torch.cat([torch.flatten(image[0])[...,None] for image in feature_space],axis=0).view(feature_space.shape[0],1,-1) # This assumes one output channel from CNN
        print(lstm_in)
        # print('the lstm input has shape ', lstm_in.shape)
        # lstm_in = torch.cat([torch.flatten(image[0]) for image in feature_space],axis=0) # This assumes one output channel
        lstm_out, self.hidden = self.lstm(lstm_in) # assuming mini-batch of 1
        # To feed the final LSTM layer through the last layer, we need to convert the multidimensional output to
        # a single dimensional tensor.
        # print(f"lstm output is {lstm_out} with shape {lstm_out.shape}")
        dense_conversion = self.prediction_converter(lstm_out)
        dense_conversion = torch.squeeze(dense_conversion)
        # print("the dense conversions are",dense_conversion)
        # print(len(dense_conversion))

        return dense_conversion

""" The following has been reassigned to evaluate.py 
model = Network(input_size, LSTM_output_size, output_dimension)

# Justifications for MSE:
# - well-known,functional in many contexts
# - this isn't, strictly, classification. We should have smooth
#   predictions
# The case against MSE:
# - Errors will be pretty small if we are working with single years. We could multiply that by some large constant.
loss_function = nn.MSELoss()

# Perhaps use ADAM, if SGD doesn't give good results
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    for current_MRI_batch, current_classifications_batch in training_data:
        # Clear gradients
        model.zero_grad()
        predictions_of_batch = []

        # loop through by patient
        for patient_MRI, patient_prognosis in zip(current_MRI_batch,current_classifications_batch):
            # clear the LSTM hidden state after each patient
            model.hidden = model.init_hidden()
            # produce prediction for current MRI scan, and append to predictions array
            current_prediction = model(patient_MRI)
            predictions_of_batch.append(current_prediction)

        # Compute loss
        loss = loss_function(torch.tensor(predictions_of_batch), current_classifications_batch)
        loss.backward()
        optimizer.step()"""
# testing. Run random data through network to ensure that everything checks out.
if __name__ == "__main__":
    big_net = Network(1,(100,100,100),3)
    iou = torch.randn(4,1,100,100,100)
    preds = big_net(iou)
    print(preds)