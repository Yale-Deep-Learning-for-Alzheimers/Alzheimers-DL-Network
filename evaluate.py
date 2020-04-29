""" Unified home for training and evaluation. Imports model and dataloader."""
# MISSING:
# * Data Importation
#
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import our network
import sys
sys.path.insert(1, '/model')
import network.py
import data_loader.py
# torch.manual_seed(314159265368979323846264338327950288419716939937510) # for reproducibility for testing purposes. Delete during actual training.

# ============= Hyperparameters ===================
batch_size = 64
# Dimensionality of the data outputted by the LSTM,
# forwarded to the final dense layer. THIS IS A GUESS CURRENTLY.
LSTM_output_size = 16
input_size = 1000 # Size of the processed MRI scans fed into the CNN. TODO: Change
output_dimension = 1 # the number of predictions the model will make
learning_rate = 0.1
training_epochs = 10
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
test_data = ...


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


# training function
def train(model,training_data,optimizer,criterion):
    """ takes (model, training data, optimizer, loss function)"""
    # activate training mode
    model.train()
    # initialize the per epoch loss
    epoch_loss = 0
    for current_MRI_batch, current_classifications_batch in training_data:
        # Clear gradients
        model.zero_grad()
        predictions_of_batch = []
        # loop through by patient
        for patient_MRI, patient_prognosis in zip(current_MRI_batch, current_classifications_batch):
            # clear the LSTM hidden state after each patient
            model.hidden = model.init_hidden()
            # produce prediction for current MRI scan, and append to predictions array
            current_prediction = model(patient_MRI)
            predictions_of_batch.append(current_prediction)
        # Compute loss
        loss = criterion(torch.tensor(predictions_of_batch), current_classifications_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss/len(training_data)

def test(model, test_data, criterion):
    """takes (model, test_data, loss function) and returns the epoch loss."""
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for current_MRI_batch, current_classifications_batch in test_data:
            # Clear gradients
            model.zero_grad()
            predictions_of_batch = []
            # loop through by patient
            for patient_MRI, patient_prognosis in zip(current_MRI_batch, current_classifications_batch):
                # clear the LSTM hidden state after each patient
                model.hidden = model.init_hidden()
                # produce prediction for current MRI scan, and append to predictions array
                current_prediction = model(patient_MRI)
                predictions_of_batch.append(current_prediction)
            # Compute loss
            loss = criterion(torch.tensor(predictions_of_batch), current_classifications_batch)
            epoch_loss += loss.item()
    return epoch_loss/len(test_data)

# perform training and measure test accuracy. Save best performing model.
best_test_accuracy = float('inf')

# this evaluation workflow was adapted from Ben Trevett's design on https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
for epoch in range(training_epochs):

    start_time = time.time()

    train_loss = train(model, training_data, optimizer, loss_function)
    test_loss = test(model, test_data, loss_function)

    end_time = time.time()

    epoch_mins = (end_time-start_time)/60
    epoch_secs = (end_time-start_time)%60

    if test_loss<best_test_accuracy:
        best_test_accuracy=test_loss
        torch.save(model.state_dict(),'ad-model.pt')

    print(f"Hurrah! Epoch {epoch+1}/{training_epochs} concludes. | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f}| Train Perplexity: {math.exp(train_loss):7.3f}")
    print(f"\tTest Loss: {test_loss:.3f}| Test Perplexity: {math.exp(test_loss):7.3f}")





