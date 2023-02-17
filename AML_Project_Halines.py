#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
""" AML_Project_Halines.py
Takes Halpha line data and uses MLP regressor and classifier to
sort lines into different bins based on their shape
"""

__author__ = "Mark Suffak"
__contact__ = "msuffak@uwo.ca"
__date__ = "2023/02/22/"
__email__ =  "msuffak@uwo.ca"
__version__ = "0.0.1"

import numpy as np
import matplotlib.pyplot as plt
import os

# Gather all file names containing Halpha line data
Halpha_files = os.listdir('Halpha_text_files')
# define needed lists
train_flux_table = []
line_type_floats = [] # line identifier for regression
line_type_strings = [] # line identifier for classification
# Loop over all files and collect data into lists
for j in range(len(Halpha_files)):
    modnumber = int(Halpha_files[j].split('mod')[1].split('_')[0]) # model number
    # open file and read first line
    file = open('Halpha_text_files/' + Halpha_files[j],'r')
    header = file.readline()
    line_type = header.split(',')[1].split('\n')[0] # double or single peaked line
    file.close()
    data = np.loadtxt('Halpha_text_files/' + Halpha_files[j], skiprows=2, delimiter=',') # get all flux v. wavelength data
    normfluxtable = data[np.where((data[:,0]<500) & (data[:,0] > -500))][:,1] # select desired data

    # Depending on line type and maximum flux, assign strings for classification, and floats for regression
    if line_type == 'Double Peaked':
        if data[np.where(data[:,0] == 1.3700737744144176),1] < 0.99:
            line_type_strings.append('Shell')
        elif max(normfluxtable) <= 10:
            line_type_strings.append('DP lt10')
        else:
            line_type_strings.append('DP gt10')
        line_type_floats.append([np.mean(normfluxtable)+np.max(normfluxtable), 0])
    elif line_type == 'Singly Peaked':
        if max(normfluxtable) <= 10:
            line_type_strings.append('SP lt10')
        else:
            line_type_strings.append('SP gt10')
        line_type_floats.append([np.mean(normfluxtable)+np.max(normfluxtable), 1])
    train_flux_table.append(normfluxtable)
# Convert list into array
train_flux_table = np.asarray(train_flux_table)

# import desired modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(train_flux_table, line_type_floats, test_size=0.2)

# Define neural network regressor
mlp = MLPRegressor(hidden_layer_sizes=(50,50), solver='adam',verbose=True)

# Fit training data to neural network
mlp.fit(train_data, train_labels)

# Feed testing data into neural network
pred_labels = mlp.predict(test_data)

# Sort the predicted labels by the first label (maximum flux)
pred_labels_sorted = sorted(pred_labels, key=lambda x: x[0])
print(pred_labels)
print(pred_labels_sorted)

# Define bins of 100 lines by first predicted label (maximum flux value)
bins = []
bins.append(pred_labels_sorted[0])
counter = 0
for i in range(len(pred_labels_sorted)):
    counter += 1
    if counter == 100:
        bins.append((pred_labels_sorted[i] + pred_labels_sorted[i+1])/2)
        counter = 0
    if i == len(pred_labels_sorted) - 1:
        bins.append(pred_labels_sorted[i])
print(len(bins), bins)

# Make directory for figures if it doesn't exist yet
if not os.path.exists('AML_regression_figs/'):
    os.mkdir('AML_regression_figs/')

# Define another set of bins to sort first set of bins by second label (single vs double peaked)
for i in range(len(bins)-1):
    bin_labels = []
    peak_labels = []
    for j in range(len(pred_labels)):
        if bins[i][0] <= pred_labels[j][0] <= bins[i + 1][0]:
            peak_labels.append(pred_labels[j][1])
    peak_bins = np.linspace(min(peak_labels), max(peak_labels),5) # define 4 bins of peak values in each bin of max flux
    print(i, peak_bins)
    # Plot data in each bin in its own figure
    for k in range(len(peak_bins)-1):
        fig, ax = plt.subplots()
        counter = 0
        for m in range(len(pred_labels)):
            if peak_bins[k] <= pred_labels[m][1] <= peak_bins[k+1] and bins[i][0] <= pred_labels[m][0] <= bins[i + 1][0]:
                ax.plot(test_data[m], alpha=0.4)
                counter += 1
        ax.text(0.75,0.8, 'Counter: ' + str(counter), transform =ax.transAxes, fontsize = 14)
        plt.savefig('AML_regression_figs/Bin_'+str(i)+'_'+str(round(peak_bins[k],2))+'_'+str(round(peak_bins[k+1],2))+'.png')

### Classification

# Use classification on the same data as above

# split data into test and train sets (using string labels, not float labels)
train_data_str, test_data_str, train_labels_str, test_labels_str = \
    train_test_split(train_flux_table, line_type_strings, test_size=0.2)

# Define classifier
mlpc = MLPClassifier(hidden_layer_sizes=(100,100), solver='adam',verbose=True)

# Fit data to classifier
mlpc.fit(train_data_str, train_labels_str)

# Predict labels from classifier
pred_labels_str = mlpc.predict(test_data_str)

print(classification_report(test_labels_str, pred_labels_str))

# Make new folder for classification figures if it doesn't exist
if not os.path.exists('AML_classification_figs/'):
    os.mkdir('AML_classification_figs/')
# Create and plot confusion matrix of classification
cm = confusion_matrix(test_labels_str, pred_labels_str, labels = mlpc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlpc.classes_)
disp.plot()
plt.savefig('AML_classification_figs/confusion_matrix.png')

# Find mislabeled lines by classifier
mislabeled_indices = np.where(pred_labels_str != test_labels_str)[0]

# Plot 16 of the mislabeled lines by the classifier
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i, ax in enumerate(ax.flat):
    ax.plot(test_data_str[mislabeled_indices[i]]) # plot the image
    ax.set(xticks=[], yticks=[]) # remove ticks
    pred = pred_labels_str[mislabeled_indices[i]]
    true = test_labels_str[mislabeled_indices[i]]
    ax.set_title("P: {:s}, A: {:s}".format(pred, true))

plt.savefig('AML_classification_figs/mislabeled_images.png')
