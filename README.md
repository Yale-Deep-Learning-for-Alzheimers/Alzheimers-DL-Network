# Multistage Classification and Prognostic Prediction of Alzheimer’s Neuroimage Sequences with a Convolutional LSTM Network
A CNN-LSTM deep learning model for multistage classification and prognostic prediction of Alzheimer's MRI neuroimages.

## Abstract

Deep convolutional neural networks augmented with a recurrent LSTM mechanism offer a powerful solution for detecting, classifying, and predicting prognoses of Alzheimer’s in patients based on MRI scans.

Our project proposes to train a neural network with CNN-LSTM architecture on MRI neuroimage data of Alzheimer’s patients to (1) detect and classify stages of Alzheimer’s and (2) yield predictive prognoses of future disease progression for individual patients based on previous MRI sequencing.

## Model Architecture

Our classification and prediction network combines a Convolutional Neural Network (CNN), which compresses the MRI neuroimages by extracting learned features; and a single Long Short Term Memory (LSTM) cell, which combines the extracted features with those of previously inputted MRI neuroimages. The output of the LSTM is fed through a single fully connected layer, to translate the multidimensional LSTM output into a single probability between 0 and 1. This network will be trained to produce the probability that a patient will develop Alzheimer’s within the next five years, weighed against a loss function that aggregates diagnoses for individual patients over time.

<em>"Multistage Classification and Prognostic Prediction of Alzheimer’s Neuroimage Sequences with a Convolutional LSTM Network" is a project for CS 452/663: Deep Learning Theory and Applications, Spring 2020.</em>
