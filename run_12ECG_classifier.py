#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

classes = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

def run_12ECG_classifier(data,header_data,classes,modelArray):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)

    for i, cl in enumerate(classes):
        current_label[i] = modelArray[i].predict(feats_reshape)[0]
        current_score[i] = modelArray[i].predict_proba(feats_reshape)[0,0]

    #current_label[label.toarray()] = 1

    # for i in range(num_classes):
    #     current_score[i] = np.array(score.toarray()[0][i])

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk
    modelArray = []
    for cl in classes:
        filename = cl+'_model.sav'
        loaded_model = joblib.load(filename)
        modelArray.append(loaded_model)

    return modelArray
