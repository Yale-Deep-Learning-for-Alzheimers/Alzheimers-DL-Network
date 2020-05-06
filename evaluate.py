""" Unified home for training and evaluation. Imports model and dataloader."""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# To unpack ADNI data
import pickle
import random

# Import network
import sys
sys.path.insert(1, './model')
from network import Network
from data_loader import MRIData
import argparse


parser = argparse.ArgumentParser(description='Train and validate network.')
parser.add_argument('--disable-cuda', action='store_true', default="True",
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print("Using CUDA. : )")
else:
    args.device = torch.device('cpu')

# torch.manual_seed(314159265368979323846264338327950288419716939937510) # for reproducibility for testing purposes. Delete during actual training.
# NOTE: don't change the seed numbers as we debug, or we might introduce user bias into the model!
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
data_shape = (200,200,150)
# Other hyperparameters unlisted: the depth of the model, the kernel size, the padding, the channel restriction.

# =========== Data import ==============
MRI_images_list = pickle.load(open("./data/Combined_MRI_List.pkl", "rb"))
# >>>>>>> 7b5e6ff7a0ab6f2c469e74a1c70a84a7bda68b4a
random.shuffle(MRI_images_list)

# How much of the data will be reserved for testing?
train_size = int(0.7 * len(MRI_images_list))

# Split list
training_list = MRI_images_list[:train_size]
test_list =  MRI_images_list[train_size:]

# print(MRI_images_list)

DATA_ROOT_DIR = './'
train_dataset = MRIData(DATA_ROOT_DIR, training_list)
test_dataset = MRIData(DATA_ROOT_DIR, test_list)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

training_data = train_loader
test_data = test_loader


# ================== Define Model =========================================
model = Network(input_size, data_shape, output_dimension).to(args.device)

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
    epoch_length = len(training_data)

    for i, patient_data in enumerate(training_data):
        if i%(math.floor(epoch_length/5)+1)==0: print(f"\t\tTesting Progress:{i/epoch_length*100}%")
        # Clear gradients
        model.zero_grad()
        # clear the LSTM hidden state after each patient
        # print("Well, the model.hidden is",model.hidden)
        model.hidden = model.init_hidden()
        print("Patient data is ",patient_data)
        #get the MRI's and classifications for the current patient
        patient_MRI = patient_data["images"]
        patient_MRI = patient_MRI.to(device=args.device)
        # print(patient_MRI.shape)
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
    epoch_length = len(test_data)
    for i, patient_data in enumerate(test_data):
        if i%(math.floor(epoch_length/5)+1)==0: print(f"\t\tTesting Progress:{i/epoch_length*100}%")
        # Clear gradients
        model.zero_grad()
        # clear the LSTM hidden state after each patient
        model.hidden = model.init_hidden()


        # get the MRI's and classifications for the current patient
        nonzero = patient_data["num_images"] # the number of images before padding starts
        patient_MRI = patient_data["images"][:nonzero] # extract actual MRI
        patient_classifications = patient_data["label"][:nonzero] # and classifications

        patient_MRI = patient_MRI.to(device=args.device) # send to CUDA, if it exists.

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
    test_loss = test(model, test_data, loss_function)

    end_time = time.time()

    epoch_mins = math.floor((end_time-start_time)/60)
    epoch_secs = math.floor((end_time-start_time)%60)

    print(f"Hurrah! Epoch {epoch + 1}/{training_epochs} concludes. | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f}| Train Perplexity: {math.exp(train_loss):7.3f}")
    print(f"\tTest Loss: {test_loss:.3f}| Test Perplexity: {math.exp(test_loss):7.3f}")


    if test_loss<best_test_accuracy:
        print("...that was our best test accuracy yet!")
        best_test_accuracy=test_loss
        torch.save(model.state_dict(),'ad-model.pt')


