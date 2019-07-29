
import numpy as np
import tensorflow as tf
import random as rn
import os
import random
import statistics as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def get_variables_cluster():
    # Reading tokenized notes from panda files
    df = pd.read_hdf('//home//srajendr//PandaFiles//tokenized_notes.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_w2v.h5')
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_GNV.h5')
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('//home//srajendr//PickleFiles//word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('//home//srajendr//PickleFiles//max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading tokenized notes eff from panda files
    df = pd.read_hdf('//home//srajendr//PandaFiles//tokenized_notes_eff.h5')
    notes_eff = df.values.tolist()

    # Reading word2vec word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_w2v_eff.h5')
    embedding_matrix_w2v_eff = df.to_numpy()

    # Reading Google word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_GNV_eff.h5')
    embedding_matrix_GNV_eff = df.to_numpy()

    # Reading word index eff from Pickle
    f = open('//home//srajendr//PickleFiles//word_index_eff.pckl', 'rb')
    word_index_eff = pickle.load(f)
    f.close()

    # Reading max length eff from Pickle
    f = open('//home//srajendr//PickleFiles//max_len_eff.pckl', 'rb')
    max_len_eff = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('//home//srajendr//PickleFiles//binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('//home//srajendr//PickleFiles//categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels

def get_variables_local():
    # Reading tokenized notes from panda files
    df = pd.read_hdf('PandaFiles/tokenized_notes.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_w2v.h5')
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_GNV.h5')
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('PickleFiles/word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('PickleFiles/max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading tokenized notes eff from panda files
    df = pd.read_hdf('PandaFiles/tokenized_notes_eff.h5')
    notes_eff = df.values.tolist()

    # Reading word2vec word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_w2v_eff.h5')
    embedding_matrix_w2v_eff = df.to_numpy()

    # Reading Google word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_GNV_eff.h5')
    embedding_matrix_GNV_eff = df.to_numpy()

    # Reading word index eff from Pickle
    f = open('PickleFiles/word_index_eff.pckl', 'rb')
    word_index_eff = pickle.load(f)
    f.close()

    # Reading max length eff from Pickle
    f = open('PickleFiles/max_len_eff.pckl', 'rb')
    max_len_eff = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('PickleFiles/binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('PickleFiles/categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels

def make_whole_labels():
    # Binary Labels
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(notes, binary_labels, test_size=0.33, random_state=39)
    X_train_b = np.array(X_train_b)
    X_test_b = np.array(X_test_b)

    # Categorical Labels
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(notes, categorical_labels, test_size=0.33, random_state=39)
    X_train_c = np.array(X_train_c)
    X_test_c = np.array(X_test_c)

    return X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c

def make_eff_labels():
    # Binary Labels
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(notes_eff, binary_labels, test_size=0.33, random_state=39)
    X_train_b = np.array(X_train_b)
    X_test_b = np.array(X_test_b)

    # Categorical Labels
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(notes_eff, categorical_labels, test_size=0.33, random_state=39)
    X_train_c = np.array(X_train_c)
    X_test_c = np.array(X_test_c)

    return X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c


# Choose which setting you are running the model in and comment one othe two next lines out
notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables_cluster()
#notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables_local()

X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_eff_labels()
#X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_whole_labels()

# temporary for local testing:
# X_train_b = X_train_b[:10]
# y_train_b = y_train_b[:10]
# X_test_b = X_test_b[:][:10]
# y_test_b = y_test_b[:][:10]

# X_train_c = X_train_c[:10]
# y_train_c = y_train_c[:10]
# X_test_c = X_test_c[:][:10]
# y_test_c = y_test_c[:][:10]


# create a plot for the model
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    del fig
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def evaluate_model(metrics, categorical, model, y_test, X_test):
    y_pred=model.predict(X_test,verbose=1)
    if (categorical): ##Check this out, weird##
        y_pred_coded = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        metric=[]
        metric.append(['f1score',f1_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['precision',precision_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['recall',recall_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
        print(metric)
        metrics.append(metric)
    else:
        y_pred_coded=np.where(y_pred>0.5,1,0)
        y_pred_coded=y_pred_coded.flatten()
        metric=[]
        metric.append(['f1score',f1_score(y_test,y_pred_coded)])
        metric.append(['precision',precision_score(y_test,y_pred_coded)])
        metric.append(['recall',recall_score(y_test,y_pred_coded)])
        metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
        print(metric)
        metrics.append(metric)
    
    return metrics, y_pred

#################################################################################################

# Create CNN model
def CNN_model(word_index, embedding_matrix, max_len, categorical, dropout, multiplier):
    model = Sequential()
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D((128*multiplier), 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout))
    if (categorical):
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
    
    return model

# CNN Model Creation 2
def CNN_model2(word_index, embedding_matrix, max_len, categorical, dropout, multiplier):
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D((64*multiplier), 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D((64*multiplier), 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense((32*multiplier), activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(dropout))
    if (categorical):
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])

    return model

#################################################################################################

# CNN Model
def CNN(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed, categorical, dropout, multiplier):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto', restore_best_weights=True) # pateince is number of epochs
    callbacks_list = [earlystop]
    if (categorical):
        kfold = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    else:
        kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    model_infos = []
    metrics = []
    model = None
    for i,(train, test) in enumerate(kfold):
        model = None
        model = CNN_model(word_index, embedding_matrix, max_len, categorical, dropout, multiplier)
        #model = CNN_model2(word_index, embedding_matrix, max_len, categorical, dropout, multiplier)
        print("Fit fold", i+1," ==========================================================================")
        model_info=model.fit(X_train[train], y_train[train], epochs=10, batch_size=4, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=0)
        print("Performance plot of fold {}:".format(i+1))
        # summarize history in plot
        plot_model_history(model_info)
        model_infos.append(model_info)

        #Final evaluation of the model
        metrics, y_pred = evaluate_model(metrics, categorical, model, y_test, X_test)
    
    print(model.summary())
    
    return y_pred, metrics, model_infos

########################################################################################################

def findAverage(all_metrics):
    f1 = []
    precision = []
    recall = []
    accuracy = []
    avg_metrics = []
    for metrics in all_metrics:
        print(metrics)
        f1.append(metrics[0][1])
        precision.append(metrics[1][1])
        recall.append(metrics[2][1])
        accuracy.append(metrics[3][1])
    avg_metrics.append(['f1score',np.mean(f1)])
    avg_metrics.append(['precision',np.mean(precision)])
    avg_metrics.append(['recall',np.mean(recall)])
    avg_metrics.append(['accuracy',np.mean(accuracy)])
    
    return avg_metrics

seed = 97

dropouts = [0.1, 0.2, 0.4]
multipliers = [1, 2]
best_dropout = 0
best_multiplier = 0
best_accuracy = 0

# Grid Search Based on Accuracy for CNN Model
for dropout in dropouts:
    for multiplier in multipliers:
        y_pred, metrics, model_infos = CNN(X_train_c, y_train_c, X_test_c, y_test_c, word_index_eff, embedding_matrix_w2v_eff, max_len_eff, seed, True, dropout, multiplier)
        avg_metrics = findAverage(metrics)
        print("Average Scores for Dropout " + str(dropout) + " and Multiplier " + str(multiplier))
        print(avg_metrics)
        if avg_metrics[3][1] > best_accuracy:
            best_accuracy = avg_metrics[3][1]
            best_dropout = dropout
            best_multiplier = multiplier

print("Best Accuracy")
print(best_accuracy)
print("Best Dropout")
print(best_dropout)
print("Best Multiplier")
print(best_multiplier)







