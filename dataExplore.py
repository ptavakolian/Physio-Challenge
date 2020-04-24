# -*- coding: utf-8 -*-
# Based on https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder = r'D:\PhysioNet'
outFolder = os.path.join(folder, 'results')
dataRecords = glob.glob(folder + '\Training_WFDB\*.mat')

doMakeDatabase = False

labelTypes = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

def encodeOneHot(labels):
    labelOneHot = [0] * len(labelTypes)
    for i, lab in enumerate(labelTypes):
        if lab in labels:
            labelOneHot[i] = 1
    return labelOneHot

if doMakeDatabase:
    for i, d in enumerate(dataRecords):
        patient = os.path.basename(d)[:-4]
        # Read associated header file to get labels
        with open(os.path.join(os.path.dirname(d), patient + '.hea')) as f:
            for line in f:
                if line.split()[0] == '#Dx:':
                    labels = [lab for lab in line.split()[1].split(',')]
                    labelOneHot = encodeOneHot(labels)
        # Place the resulting labels in a data frame
        labelDict = dict(zip(labelTypes, labelOneHot))
        if i == 0:
            dataDF = pd.DataFrame(labelDict, index=[i])
        else:
            tempDF = pd.DataFrame(labelDict, index=[i])
            dataDF = dataDF.append(tempDF)

    # Save dataDF to file
    dataDF.to_csv('D:\PhysioNet\Training_WFDB\dataLabels.csv', index=False)
else:
    dataDF = pd.read_csv('D:\PhysioNet\Training_WFDB\dataLabels.csv')

# Make histogram for individual labels
categories = list(dataDF.columns.values)
sns.set(font_scale = 2)
plt.figure(figsize=(15,8))
ax= sns.barplot(categories, dataDF.sum().values)
plt.title("Comments in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Comment Type ', fontsize=18)
#adding the text labels
rects = ax.patches
labels = dataDF.sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
plt.savefig(os.path.join(outFolder,'LabelsHist.png'))
plt.show()

# Make histogram of instances with multuiple labels
rowSums = dataDF.sum(axis=1)
multiLabel_counts = rowSums.value_counts()
sns.set(font_scale = 2)
plt.figure(figsize=(15,8))
ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Comments having multiple labels ")
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)
#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.savefig(os.path.join(outFolder,'MultiLabelsHist.png'))
plt.show()