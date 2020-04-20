""" Barebones pytorch implementation of a CNN LSTM, modified from a pytorch tutorial
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) # probably change this


# ============= Hyperparameters ===================
batch_size = 64
LSTM_output_size = 16 # Dimensionality of the data outputted by the LSTM, forwarded to the final dense layer. THIS IS A GUESS CURRENTLY
# TODO: Find a sensible output size for the LSTM, perhaps by reviewing literature.
input_size = 1000 # Size of the processed MRI scans fed into the CNN. TODO: Change

output_dimension = 1 #TODO: Formally decide what output we want. I suggest having a single number: the probability (0-1) of developing Alzheimer's in the next (5) years.
learning_rate = 0.1
training_epochs = 300


class Network(nn.Module):
    """ CNN LSTM to classify ADNI data """

    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(Network, self).__init__()
        self.hidden_dim = hidden_dim

        # CNN for feature selection and encoding.
        self.convolution1 = nn.Conv3d(embedding_dim, 400) #TODO: Optimize ALL of these when Data Size is known
        self.pool1 = nn.MaxPool3d(...)
        self.convolution2 = nn.Conv3d(200, 100)
        self.pool2 = nn.MaxPool3d(...)

        # LSTM to combine feature encoding from above with feature encodings from past networks
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.prediction_converter = nn.Linear(hidden_dim, output_size)

    def forward(self, MRI):
        feature_space = self.pool2(self.convolution2(self.pool1(self.convolution1(MRI))))
        lstm_out, _ = self.lstm(feature_space)
        prediction_given = self.prediction_converter(lstm_out.view(len(sentence), -1))
        prediction = F.log_softmax(tag_space, dim=1)
        return prediction


model = Network(input_size, LSTM_output_size, output_dimension)

loss_function = nn.NLLLoss() # I believe this should work. We'll have

optimizer = optim.SGD(model.parameters(), lr=learning_rate) #TODO: Should we use Adam?

for epoch in range(training_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    for MRI, tags in training_data:
        # Clear gradients
        model.zero_grad()

        # Clear LSTM Hidden State after each batch.
        model.hidden = model.init_hidden()

        # DATA PROCESSING FUNCTIONS HERE
        MRI = ...

        prediction = model(MRI)

        # Compute loss
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

#TODO: create specialized score from tagging data.

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    print(tag_scores)