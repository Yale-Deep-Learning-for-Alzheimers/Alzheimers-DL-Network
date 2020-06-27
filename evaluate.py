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
parser.add_argument('--disable-cuda', action='store_true', default=False,
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
print(args.disable_cuda)
if torch.cuda.is_available():
    print("Using CUDA. : )")
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args.device = torch.device('cuda')
else:
    print("We aren't using CUDA.")
    args.device = torch.device('cpu')

# For reproducibility for testing purposes. Delete during actual training.
# torch.manual_seed(1)
# random.seed(1)

## Hyperparameters

BATCH_SIZE = 10
# Dimensionality of the data outputted by the LSTM,
# forwarded to the final dense layer.
LSTM_output_size = 16
input_size = 1 # Size of the processed MRI scans fed into the CNN.

output_dimension = 2 # the number of predictions the model will make
# 2 used for binary prediction for each image.
# update the splicing used in train()

learning_rate = 0.1
training_epochs = 5
# The size of images passed, as a tuple
data_shape = (200,200,150)
# Other hyperparameters unlisted: the depth of the model, the kernel size, the padding, the channel restriction.


## Import Data
MRI_images_list = pickle.load(open("./Data/Combined_MRI_List.pkl", "rb"))
random.shuffle(MRI_images_list)

train_size = int(0.7 * len(MRI_images_list))

# Split list
training_list = MRI_images_list[:train_size]
test_list =  MRI_images_list[train_size:]

DATA_ROOT_DIR = './'
train_dataset = MRIData(DATA_ROOT_DIR, training_list)
test_dataset = MRIData(DATA_ROOT_DIR, test_list)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

training_data = train_loader
test_data = test_loader


## Define Model
model = Network(input_size, data_shape, output_dimension).to(args.device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)


## Training Function
def train(model,training_data,optimizer,criterion):
    """ takes (model, training data, optimizer, loss function)"""
    # Activate training mode
    model.train()
    # Initialize the per epoch loss
    epoch_loss = 0
    epoch_length = len(training_data)
    for i, patient_data in enumerate(training_data):
        if i % (math.floor(epoch_length / 5) + 1) == 0: print(f"\t\tTraining Progress:{i / len(training_data) * 100}%")
        # Clear gradients
        model.zero_grad()
        torch.cuda.empty_cache() # Clear CUDA memory
        batch_loss=torch.tensor(0.0).to(args.device)

        # Clear the LSTM hidden state after each patient
        model.hidden = model.init_hidden()

        # Get the MRI's and classifications for the current patient
        patient_markers = patient_data['num_images']
        patient_MRIs = patient_data["images"].to(args.device)

        patient_classifications = patient_data["label"]
        print("Patient batch classes ", patient_classifications)

        for x in range(len(patient_MRIs)):
            try:
                # Clear hidden states to give each patient a clean slate
                model.hidden = model.init_hidden()
                single_patient_MRIs = patient_MRIs[x][:patient_markers[x]].view(-1,1,data_shape[0],data_shape[1],data_shape[2])

                patient_diagnosis = patient_classifications[x]
                patient_endstate = torch.ones(single_patient_MRIs.size(0)) * patient_diagnosis
                patient_endstate = patient_endstate.long().to(args.device)

                out = model(single_patient_MRIs)

                if len(out.shape)==1:
                    out = out[None,...] # In the case of a single input, we need padding

                print("model predictions are ",out)
                print("patient endstate is ",patient_endstate)
                model_predictions = out

                loss = criterion(model_predictions, patient_endstate)
                batch_loss += loss

            except Exception as e:
                print("EXCEPTION CAUGHT:",e)

        batch_loss.backward()
        print("batch loss is",batch_loss)
        optimizer.step()
        epoch_loss += batch_loss

    if epoch_length == 0: epoch_length = 0.000001
    return epoch_loss / epoch_length


## Testing Function
def test(model, test_data, criterion):
    """takes (model, test_data, loss function) and returns the epoch loss."""
    model.eval()
    epoch_loss = torch.tensor(0.0)
    epoch_length = len(test_data)
    for i, patient_data in enumerate(test_data):
        if i % (math.floor(epoch_length / 5) + 1) == 0: print(f"\t\tTesting Progress:{i / len(test_data) * 100}%")
        # Clear gradients
        model.zero_grad()
        torch.cuda.empty_cache() # Clear CUDA memory

        # Clear the LSTM hidden state after each patient
        model.hidden = model.init_hidden()
        # Get the MRI's and classifications for the current patient
        patient_markers = patient_data['num_images']
        patient_MRIs = patient_data["images"].to(args.device)

        patient_classifications = patient_data["label"]
        print("Patient batch classes ", patient_classifications)
        for x in range(len(patient_MRIs)):
            try:
                # Clear hidden states to give each patient a clean slate
                model.hidden = model.init_hidden()
                single_patient_MRIs = patient_MRIs[x][:patient_markers[x]].view(-1, 1, data_shape[0], data_shape[1],
                                                                                data_shape[2])
                single_patient_MRIs = single_patient_MRIs
                patient_diagnosis = patient_classifications[x]
                patient_endstate = torch.ones(single_patient_MRIs.size(0)) * patient_diagnosis
                patient_endstate = patient_endstate.long().to(args.device)

                out = model(single_patient_MRIs)

                if len(out.shape)==1:
                    out = out[None,...] # In the case of a single input, we need padding

                model_predictions = out

                loss = criterion(model_predictions, patient_endstate)
                epoch_loss += loss
                print("Current test loss ",loss)
            except Exception as e:
                epoch_length -= 1
                print("EXCEPTION CAUGHT:", e)

    if epoch_length == 0: epoch_length = 0.000001
    return epoch_loss / epoch_length

# Perform training and measure test accuracy. Save best performing model.
best_test_accuracy = float('inf')

# This evaluation workflow below was adapted from Ben Trevett's design
# on https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
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
