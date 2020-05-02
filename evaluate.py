""" Unified home for training and evaluation. Imports model and dataloader."""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# To unpack ADNI data
import pickle
import random

# Import network
import sys
sys.path.insert(1, './model')
from network import Network
from data_loader import MRIData
# torch.manual_seed(314159265368979323846264338327950288419716939937510) # for reproducibility for testing purposes. Delete during actual training.
# NOTE: don't change the seed numbers as we debug --- the specific files in data_sample are dependent on these seeds
torch.manual_seed(1)
random.seed(1)

# ============= Hyperparameters ===================
BATCH_SIZE = 64
# Dimensionality of the data outputted by the LSTM,
# forwarded to the final dense layer. THIS IS A GUESS CURRENTLY.
LSTM_output_size = 16
input_size = 3 # Size of the processed MRI scans fed into the CNN.

output_dimension = 4 # the number of predictions the model will make
# 2 are converted into a binary diagnosis, two are used for prediction
# NOTE: The training architecture currently assumes 4. If you change output-dimension,
# update the splicing used in train()

learning_rate = 0.1
training_epochs = 10
# The size of images passed, as a tuple
data_shape = (192,192,160)
# Other hyperparameters unlisted: the depth of the model, the kernel size, the padding, the channel restriction.


# ========== TODO: Import Data ==============
# expected format:
# training_data stores batches of MRI's and classifications like this: [batch,batch,batch] : )
# each batch should be in form
# [Bunch of MRIs, Bunch of Classifications]
# and each 'bunch' in the batch should be grouped by patient
# Bunch of MRIs = [Patient 1 MRIs, Patient 2 MRIs,...]
# Bunch of Classifications = [Patient 1 classifications, Patient 2 Classifications...]
# the Classifications should be binary 0,1 probabilities in output_dimension dimensions. Perhaps something like this:
# [chance_of_normality: 0 , chance of MCI: 0, chance of AD: 1]

MRI_images_list = pickle.load(open("./data/Combined_MRI_List.pkl", "rb"))
random.shuffle(MRI_images_list)
# NOTE: simply for testing out the data loader, take the first three images from the list
MRI_images_list = MRI_images_list[:3] # these 3 are in data_sample folder
                                      # root dir should be './data_sample/'
# print(MRI_images_list)

# ========== TODO: Use DataLoader to Create Train/Test Split ==============

DATA_ROOT_DIR = './data_sample'
train_dataset = MRIData(DATA_ROOT_DIR, MRI_images_list)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
"""
for it, train_data in enumerate(train_loader):
    print("Test\n")
    print(train_data)
    print("The first index is ",train_data['images'][0])
    print("classifications are ",train_data['label'])
    print(train_data['images'][0].shape)
"""
training_data = train_loader
test_data = ...


model = Network(input_size, data_shape, output_dimension)

loss_function = nn.CrossEntropyLoss()

# Perhaps use ADAM, if SGD doesn't give good results
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# training function
def train(model,training_data,optimizer,criterion):
    """ takes (model, training data, optimizer, loss function)"""
    # activate training mode
    model.train()
    # initialize the per epoch loss
    epoch_loss = 0
    # TODO: is enumerate necessary? Maybe build a progress function into the thing.
    for i, patient_data in enumerate(training_data):
        # Clear gradients
        model.zero_grad()
        # clear the LSTM hidden state after each patient
        # TODO: Figure out if we need to clear the LSTM hidden states inbetween patients, and if so how to do this.
        # print("Well, the model.hidden is",model.hidden)
        # model.hidden = model.init_hidden(len(patient_data))

        #get the MRI's and classifications for the current patient
        patient_MRI = patient_data["images"]
        patient_classifications = patient_data["label"]
        # print("patient classes ", patient_classifications)
        patient_endstate = torch.ones(len(patient_classifications)) * patient_classifications[-1]
        patient_endstate = patient_endstate.long()

        # print("The number of input channels appears to be", patient_MRI.shape)

        # produce prediction for current MRI scan, and append to predictions array
        out = model(patient_MRI)
        # print("model gives ",out)
        #loss = criterion(out,patient_super)

        # loss from model diagnoses
        model_diagnoses = out[:,:2]
        # print("model diagnosis is ",model_diagnoses)# extract the first two columns of the output, which we train classify the MRIs
        # Compute loss with respect to
        loss = criterion(model_diagnoses, patient_classifications)

        # for model prediction and classification
        model_predictions = out[:,2:] # extract the second two columns of the output
        # print("model predictions are ",model_predictions)
        loss += criterion(model_predictions, patient_endstate)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    return epoch_loss/len(training_data)

def test(model, test_data, criterion):
    """takes (model, test_data, loss function) and returns the epoch loss."""
    model.eval()
    epoch_loss = 0
    for i, patient_data in enumerate(test_data):
        # Clear gradients
        model.zero_grad()
        # clear the LSTM hidden state after each patient
        # print("Well, the model.hidden is",model.hidden)
        # model.hidden = model.init_hidden(len(patient_data))

        # get the MRI's and classifications for the current patient
        patient_MRI = patient_data["images"]
        patient_classifications = patient_data["label"]
        # print("patient classes ", patient_classifications)
        patient_endstate = torch.ones(len(patient_classifications)) * patient_classifications[-1]
        patient_endstate = patient_endstate.long()

        # print("The number of input channels appears to be", patient_MRI.shape)

        # produce prediction for current MRI scan, and append to predictions array
        out = model(patient_MRI)
        # print("model gives ",out)
        # loss = criterion(out,patient_super)

        # loss from model diagnoses
        model_diagnoses = out[:, :2]
        # print("model diagnosis is ",model_diagnoses)# extract the first two columns of the output, which we train classify the MRIs
        # Compute loss with respect to
        loss = criterion(model_diagnoses, patient_classifications)

        # for model prediction and classification
        model_predictions = out[:, 2:]  # extract the second two columns of the output
        # print("model predictions are ",model_predictions)
        loss += criterion(model_predictions, patient_endstate)

        epoch_loss += loss.item()

    return epoch_loss/len(test_data)

# perform training and measure test accuracy. Save best performing model.
best_test_accuracy = float('inf')

# this evaluation workflow was adapted from Ben Trevett's design on https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
for epoch in range(training_epochs):

    start_time = time.time()

    train_loss = train(model, training_data, optimizer, loss_function)
    test_loss = 0 #test(model, test_data, loss_function)

    end_time = time.time()

    epoch_mins = math.floor((end_time-start_time)/60)
    epoch_secs = math.floor((end_time-start_time)%60)

    if test_loss<best_test_accuracy:
        best_test_accuracy=test_loss
        torch.save(model.state_dict(),'ad-model.pt')

    print(f"Hurrah! Epoch {epoch+1}/{training_epochs} concludes. | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f}| Train Perplexity: {math.exp(train_loss):7.3f}")
    print(f"\tTest Loss: {test_loss:.3f}| Test Perplexity: {math.exp(test_loss):7.3f}")
