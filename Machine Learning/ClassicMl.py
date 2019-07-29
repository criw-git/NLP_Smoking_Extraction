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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_variables_local():
    f = open('PickleFiles/cleaned_notes.pckl', 'rb')
    cleaned_notes = pickle.load(f)
    f.close()

    # Reading tokenized notes eff from panda files
    df = pd.read_hdf('PandaFiles/tokenized_notes_eff.h5')
    notes_eff = df.values.tolist()

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

    return cleaned_notes, notes_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels

cleaned_notes, notes_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables_local()

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(cleaned_notes, binary_labels, test_size=0.33, random_state = 42)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(cleaned_notes, categorical_labels, test_size=0.33, random_state=39)
y_train_c = np.argmax(y_train_c, axis=1)
y_test_c = np.argmax(y_test_c, axis=1)

samples = 1

###############################################################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer # add reference
print("Naive Bayes Binary Classification")

for i in range(0, samples):
    nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
    nb.fit(X_train_b, y_train_b)
    y_pred = nb.predict(X_test_b)
    print('f1 Score %s' % f1_score(y_pred, y_test_b))
    print('precision %s' % precision_score(y_pred, y_test_b))
    print('recall %s' % recall_score(y_pred, y_test_b))
    print('accuracy %s' % accuracy_score(y_pred, y_test_b))
    print("---------------------------------")
    nb = None

print("####################################")
# ###############################################################################
from sklearn.linear_model import SGDClassifier
print("Binary Classification for SVM Model")

for i in range(0, samples):
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
    sgd.fit(X_train_b, y_train_b)
    y_pred = sgd.predict(X_test_b)
    print('f1 Score %s' % f1_score(y_pred, y_test_b))
    print('precision %s' % precision_score(y_pred, y_test_b))
    print('recall %s' % recall_score(y_pred, y_test_b))
    print('accuracy %s' % accuracy_score(y_pred, y_test_b))
    print("---------------------------------")
    sgd = None

print("####################################")
#################################################################################
from sklearn.linear_model import LogisticRegression
print("Binary Classification for Logistic Regression")

for i in range(0, samples):
    logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
    logreg.fit(X_train_b, y_train_b)
    y_pred = logreg.predict(X_test_b)
    print('f1 Score %s' % f1_score(y_pred, y_test_b))
    print('precision %s' % precision_score(y_pred, y_test_b))
    print('recall %s' % recall_score(y_pred, y_test_b))
    print('accuracy %s' % accuracy_score(y_pred, y_test_b))
    print("---------------------------------")
    logreg = None

print("####################################")
##################################################################################
##################################################################################
print("Non-Binary Classification for Naive Bayes Model")

for i in range(0, samples):
    nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
    nb.fit(X_train_c, y_train_c)
    y_pred = nb.predict(X_test_c)
    print('f1 Score %s' % f1_score(y_pred, y_test_c, average='weighted'))
    print('precision %s' % precision_score(y_pred, y_test_c, average='weighted'))
    print('recall %s' % recall_score(y_pred, y_test_c, average='weighted'))
    print('accuracy %s' % accuracy_score(y_pred, y_test_c))
    print("---------------------------------")
    nb = None

print("####################################")
##################################################################################
print("Non-Binary Classification for SVM Model")

for i in range(0, samples):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])
    sgd.fit(X_train_c, y_train_c)
    y_pred = sgd.predict(X_test_c)
    print('f1 Score %s' % f1_score(y_pred, y_test_c, average='weighted'))
    print('precision %s' % precision_score(y_pred, y_test_c, average='weighted'))
    print('recall %s' % recall_score(y_pred, y_test_c, average='weighted'))
    print('accuracy %s' % accuracy_score(y_pred, y_test_c))
    print("---------------------------------")
    sgd = None

print("####################################")
##################################################################################
print("Non-Binary Classification for Logistic Regression Model")

for i in range(0, samples):
    logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
    logreg.fit(X_train_c, y_train_c)
    y_pred = logreg.predict(X_test_c)
    print('f1 Score %s' % f1_score(y_pred, y_test_c, average='weighted'))
    print('precision %s' % precision_score(y_pred, y_test_c, average='weighted'))
    print('recall %s' % recall_score(y_pred, y_test_c, average='weighted'))
    print('accuracy %s' % accuracy_score(y_pred, y_test_c))
    print("---------------------------------")
    logreg = None

print("####################################")
##################################################################################