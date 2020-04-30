""" Barebones pytorch implementation of a CNN LSTM, modified from a pytorch tutorial
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html """

# Current state of the model:
# Very likely has many bugs. I need to run some data through it to catch them.
# Two current concerns:
# - The output of the LSTM layer is a big matrix, that must be condensed to 1 dimension using the View function
# - Currently, the model expects and produces probabilities (0-1) of AD diagnosis in some timeframe.
#   What sort of loss function is best for this? MSE would have a hard time with the small numbers, but Cross Entropy
#   treats it like a classification problem, which it isn't. Perhaps we could multiply the obtained probabilities
#   by some large constant to get a better error space.



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # for reproducibility for testing purposes. Delete during actual training.


# ============= Hyperparameters ===================
batch_size = 64

# Dimensionality of the data outputted by the LSTM,
# forwarded to the final dense layer. THIS IS A GUESS CURRENTLY
# TODO: Find a sensible output size for the LSTM, perhaps by reviewing literature.
LSTM_output_size = 16

input_size = 1000 # Size of the processed MRI scans fed into the CNN. TODO: Change

output_dimension = 1
learning_rate = 0.1
training_epochs = 50
# Other hyperparameters unlisted: the depth of the model, the kernel size, the padding, the channel restriction.


# ========== Import Training Data Here ==============
# expected format:
# training_data stores batches of MRI's and classifications like this:
# [[Patient MRIs,Patient Classifications],[Patient MRIs, Patient Classifications]...]
# The Patient MRIs be ...
# the Classifications are expected as single integers indicating the chance a patient will
# get Alzheimer's within the next (5) years, with a positive diagnosis given by '1'.

training_data = ...


class Network(nn.Module):
    """ CNN-LSTM to classify ADNI data """

    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(Network, self).__init__()
        self.hidden_dim = hidden_dim

        # CNN for feature selection and encoding.

        # CNN Specific Hyperparameters: TODO: Optimize! These are guesses.
        kernel_size = 3
        padding = 0

        # The input and output
        self.convolution1 = nn.Conv3d(embedding_dim, 400, kernel_size, padding=padding) #TODO: Optimize ALL of these when Data Size is known
        # We may want to increase the channel size from input to gain better performance.
        self.pool1 = nn.MaxPool3d(kernel_size)
        self.convolution2 = nn.Conv3d(200, 100, kernel_size, padding=padding)
        self.pool2 = nn.MaxPool3d(kernel_size)

        # LSTM to combine feature encoding from above with feature encodings from past networks
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to prediction space
        self.prediction_converter = nn.Linear(hidden_dim, output_size)

    def forward(self, MRI):
        feature_space = self.pool2(self.convolution2(self.pool1(self.convolution1(MRI))))
        lstm_out, _ = self.lstm(feature_space)
        # To feed the final LSTM layer through the last layer, we need to convert the multidimensional output to
        # a single dimensional tensor. # TODO: Figure out how to use this view function.
        dense_conversion = self.prediction_converter(lstm_out.view(len(lstm_out), -1))
        prediction = F.log_softmax(dense_conversion, dim=1)
        return prediction


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
        optimizer.step()
