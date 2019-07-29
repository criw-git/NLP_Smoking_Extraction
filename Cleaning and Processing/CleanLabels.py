import os
import csv
import pandas as pd
import numpy as np
import pickle
from keras.utils import np_utils

# load in all the files in each of the folders into two lists --> same order
progressNoteFiles = os.listdir('PatientData/')
smokingDataFiles = os.listdir('SmokingData/')

def get_word_labels(smokingDataFiles, root):
    status = []

    # doesn't capture the first element but don't really need it because we are looking at the last element
    for f in smokingDataFiles:
        df = pd.read_csv(root + f)
        status.append(df.values.tolist())


    word_labels = []
    for i in range(len(status)):
        if (len(status[i]) > 0):
            word_labels.append((status[i][-1][0]))
    
    #getting rid of Unknown and Never Assessed
    a = []
    for word_label in word_labels:
        if (word_label != 'Never Assessed'):
            if (word_label != 'Unknown If Ever Smoked'):
                a.append(word_label)
    word_labels = a

    return word_labels

def make_binary_labels(word_labels):
    labels = []
    for word_label in word_labels:
        if (word_label == 'Former Smoker' or word_label == 'Current Every Day Smoker' or 
                word_label == 'Current Some Day Smoker' or word_label == 'Light Tobacco Smoker' or 
                word_label == 'Heavy Tobacco Smoker' or word_label == 'Smoker, Current Status Unknown'):
                labels.append(1)
        elif (word_label == 'Never Smoker' or word_label == 'Passive Smoke Exposure - Never Smoker' or
            word_label == 'Never Assessed' or word_label == 'Unknown If Ever Smoked'):
                labels.append(0)
        else:
                labels.append(2)

    labels = np.array(labels)
    return labels

def make_categorical_labels(word_labels):
    labels = []
    for word_label in word_labels:
        if (word_label == 'Current Every Day Smoker' or word_label == 'Current Some Day Smoker' 
            or word_label == 'Light Tobacco Smoker' or word_label == 'Heavy Tobacco Smoker' 
            or word_label == 'Smoker, Current Status Unknown'):
                labels.append(2)
        if (word_label == 'Former Smoker'):
                labels.append(1)
        elif (word_label == 'Never Smoker' or word_label == 'Passive Smoke Exposure - Never Smoker' or
            word_label == 'Never Assessed' or word_label == 'Unknown If Ever Smoked'):
                labels.append(0)

    labels = np.array(labels)
    l = np_utils.to_categorical(labels)
    return l

word_labels = get_word_labels(smokingDataFiles, "SmokingData/")
binary_labels = make_binary_labels(word_labels)
categorical_labels = make_categorical_labels(word_labels)

f = open('PickleFiles/binary_labels.pckl', 'wb')
pickle.dump(binary_labels, f)
f.close()

f = open('PickleFiles/categorical_labels.pckl', 'wb')
pickle.dump(categorical_labels, f)
f.close()


