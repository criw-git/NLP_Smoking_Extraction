import matplotlib.pyplot as plt
import numpy as np

N = 6 # change to 6 and add LSTMU and LSTMB Models
font = 'x-large'
title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'}
###########################################################################
# Binary DL Graph Models
f1_means = (0.853993887, 0.811600324, 0.8017485516, 0.8069866652, 0.8024736733, 0.8004409841)
f1_err = (0.0161547238553041, 0.00414287866088785, 0.02670095987, 0.01426782532, 0.03579518095, 0.01343905201)
precision_means = (0.888551695, 0.68561981031667, 0.6881050722, 0.6924379413, 0.6919641535, 0.688190854)
precision_err = (0.0418452561710614, 0.00116344084501155, 0.006112392218, 0.005777803157, 0.01871322656, 0.004446866707)
recall_means = (0.824293785310734, 0.994350282485875, 0.9638418064, 0.9683615819, 0.9661016949, 0.9574011299)
recall_err = (0.0337357474989223, 0.0106532095094021, 0.07314147954, 0.0448195097, 0.1071958529, 0.03856936352)
accuracy_means = (0.806589147286821, 0.683333333333333, 0.6751937949, 0.6829457364, 0.6786821705, 0.6731007752)
accuracy_err = (0.0232880912958136, 0.00518408337150901, 0.02550166437, 0.01389114218, 0.0232880913, 0.01459456596)

fig, ax = plt.subplots()

ind = np.arange(N) 
width = 0.10

ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
ax.bar(ind + width, precision_means, width,
    label='Precision', yerr=precision_err)
ax.bar(ind + 2*width, recall_means, width,
    label='Recall', yerr=recall_err)
ax.bar(ind + 3*width, accuracy_means, width,
    label='Accuracy', yerr=accuracy_err)

x_labels = ['CNN Pre', 'CNN Post', 'LSTMU Pre', 'LSTMU Post', 'LSTMB Pre', 'LSTMB Post']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Models', fontsize = font)
ax.set_title('Binary DL Model Performances', **title_font)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.show()
##################################################################################
# Non-Binary DL Bar Graph
f1_means = (0.680436811, 0.397195236, 0.4134755668, 0.422475575648325, 0.5329457364, 0.501)
precision_means = (0.711238296411343, 0.40775712823729, 0.4081667731, 0.387854684047091, 0.5329457364, 0.501)
recall_means = (0.693798449612402, 0.512790697674418, 0.5313953488, 0.51860465116279, 0.5329457364, 0.501)
accuracy_means = (0.693798449612403, 0.512790697674418, 0.5313953488, 0.51860465116279, 0.5329457364, 0.501)
f1_err = (0.0399069890331443, 0.072036577433337, 0.02701856799, 0.0270013828625037, 0.02334536305, 0.05159538942)
precision_err = (0.0380529926709191, 0.0751936234383296, 0.08495896048, 0.0623189223876878, 0.02334536305, 0.05159538942)
recall_err = (0.0305194103643869, 0.0654734166653383, 0.01689001638, 0.0368069692621588, 0.02334536305, 0.05159538942)
accuracy_err = (0.0305194103643868, 0.0654734166653383, 0.01689001638, 0.0368069692621588, 0.02334536305, 0.05159538942)

fig, ax = plt.subplots()

ind = np.arange(N) 

ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
ax.bar(ind + width, precision_means, width,
    label='Precision', yerr=precision_err)
ax.bar(ind + 2*width, recall_means, width,
    label='Recall', yerr=recall_err)
ax.bar(ind + 3*width, accuracy_means, width,
    label='Accuracy', yerr=accuracy_err)

x_labels = ['CNN Pre', 'CNN Post', 'LSTMU Pre', 'LSTMU Post', 'LSTMB Pre', 'LSTMB Post']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Models', fontsize = font)
ax.set_title('Multi-class DL Model Performances', **title_font)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.show()
########################################################################
N = 3
# Binary CL Graph Models
f1_means = (0.785882352941176, 0.723032069970845, 0.725146198830409)
f1_err = (0, 0, 0)
precision_means = (1, 0.74251497005988, 0.74251497005988)
precision_err = (0, 0, 0)
recall_means = (0.647286821705426, 0.704545454545454, 0.708571428571428)
recall_err = (0, 0, 0)
accuracy_means = (0.647286821705426, 0.631782945736434, 0.635658914728682)
accuracy_err = (0, 0, 0)

fig, ax = plt.subplots()

ind = np.arange(N)

ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
ax.bar(ind + width, precision_means, width,
    label='Precision', yerr=precision_err)
ax.bar(ind + 2*width, recall_means, width,
    label='Recall', yerr=recall_err)
ax.bar(ind + 3*width, accuracy_means, width,
    label='Accuracy', yerr=accuracy_err)

x_labels = ['NB', 'SVM', 'LR']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Models', fontsize = font)
ax.set_title('Binary Classical Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()
##################################################################################
# Non-Binary CL Graph Models
f1_means = (0.696837964690222, 0.614022779305171, 0.56230920910151)
f1_err = (0, 0, 0)
precision_means = (0.971012793191342, 0.690293035849319, 0.592131254365122)
precision_err = (0, 0, 0)
recall_means = (0.550387596899224, 0.565891472868217, 0.542635658914728)
recall_err = (0, 0, 0)
accuracy_means = (0.550387596899224, 0.565891472868217, 0.542635658914728)
accuracy_err = (0, 0, 0)

fig, ax = plt.subplots()

ind = np.arange(N) 

ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
ax.bar(ind + width, precision_means, width,
    label='Precision', yerr=precision_err)
ax.bar(ind + 2*width, recall_means, width,
    label='Recall', yerr=recall_err)
ax.bar(ind + 3*width, accuracy_means, width,
    label='Accuracy', yerr=accuracy_err)

x_labels = ['NB', 'SVM', 'LR']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Models', fontsize = font)
ax.set_title('Multi-class Classical Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()
##################################################################################
# Binary Model Accuracies
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_cnn_gnv = [0.64055, 0.77237, 0.79342, 0.79146, 0.825275, 0.7989833333, 0.8301333333]
mean_cnn_w2v = [0.58796, 0.59468, 0.5722777778, 0.5698142857, 0.5943, 0.5962]
mean_lstmu_gnv = [0.59278, 0.59183, 0.57358, 0.5559714286, 0.60765, 0.625]
mean_lstmu_w2v = [0.57256, 0.58602, 0.58609, 0.5786285714, 0.5919, 0.5847, 0.607, 0.5962]
mean_lstmb_gnv = [0.59278, 0.59085, 0.59848, 0.54388, 0.5667]
mean_lstmb_w2v = [0.59373, 0.5937, 0.58032, 0.5813375, 0.59582, 0.600375, 0.59325, 0.5905, 0.6286]
plt.plot(epoch[0:len(mean_cnn_gnv)], mean_cnn_gnv, color='r', label = 'CNN Pre-trained')
plt.plot(epoch[0:len(mean_cnn_w2v)], mean_cnn_w2v, color='r', linestyle='dashed', label = 'CNN Post-trained')
plt.plot(epoch[0:len(mean_lstmu_gnv)], mean_lstmu_gnv, color='b', label = 'LSTM_Uni Pre-trained')
plt.plot(epoch[0:len(mean_lstmu_w2v)], mean_lstmu_w2v, color='b', linestyle='dashed', label = 'LSTM_Uni Post-trained')
plt.plot(epoch[0:len(mean_lstmb_gnv)], mean_lstmb_gnv, color='k', label = 'LSTM_Bi Pre-trained')
plt.plot(epoch[0:len(mean_lstmb_w2v)], mean_lstmb_w2v, color='k', linestyle='dashed', label = 'LSTM_Bi Post-trained')
plt.xlabel('Models', fontsize = font)
plt.ylabel('Validation Accuracy', fontsize = font)
plt.title('Validation Accurcay of Binary Models', **title_font)
plt.show()

##########################################################################
# Binary Model Losses
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_cnn_gnv = [0.6087, 0.38943, 0.36986, 0.37214, 0.368725, 0.3908833333, 0.4152]
mean_cnn_w2v = [0.68566, 0.67853, 0.6842333333, 0.6857571429, 0.6787, 0.6758]
mean_lstmu_gnv = [0.67946, 0.67783, 0.68354, 0.7160571429, 0.6995, 0.7016]
mean_lstmu_w2v = [0.67663, 0.67217, 0.67547, 0.6707285714, 0.65912, 0.6530333333, 0.6602, 0.68475]
mean_lstmb_gnv = [0.67891, 0.68027, 0.68545, 0.72172, 0.6924]
mean_lstmb_w2v = [0.67512, 0.67127, 0.67552, 0.6720375, 0.67872, 0.664875, 0.6568, 0.6569]
plt.plot(epoch[0:len(mean_cnn_gnv)], mean_cnn_gnv, color='r', label = 'CNN Pre-trained')
plt.plot(epoch[0:len(mean_cnn_w2v)], mean_cnn_w2v, color='r', linestyle='dashed', label = 'CNN Post-trained')
plt.plot(epoch[0:len(mean_lstmu_gnv)], mean_lstmu_gnv, color='b', label = 'LSTM_Uni Pre-trained')
plt.plot(epoch[0:len(mean_lstmu_w2v)], mean_lstmu_w2v, color='b', linestyle='dashed', label = 'LSTM_Uni Post-trained')
plt.plot(epoch[0:len(mean_lstmb_gnv)], mean_lstmb_gnv, color='k', label = 'LSTM_Bi Pre-trained')
plt.plot(epoch[0:len(mean_lstmb_w2v)], mean_lstmb_w2v, color='k', linestyle='dashed', label = 'LSTM_Bi Post-trained')
plt.xlabel('Models', fontsize = font)
plt.ylabel('Validation Losses', fontsize = font)
plt.title('Validation Losses of Binary Models', **title_font)
plt.show()
########################################################################
# Non-Binary Model Accuracies
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_cnn_gnv = [0.54895, 0.67026, 0.70169, 0.71043, 0.7130375, 0.7592571429, 0.748225, 0.7157666667]
mean_cnn_w2v = [0.41598, 0.43175, 0.43496, 0.4342666667, 0.4330714286, 0.4952, 0.5524, 0.381]
mean_lstmu_gnv = [0.44072, 0.46743, 0.45803, 0.4728285714, 0.4650666667, 0.4571]
mean_lstmu_w2v = [0.43032, 0.46284, 0.47806, 0.5143, 0.5]
mean_lstmb_gnv = [0.45025, 0.47037, 0.46091, 0.4621777778, 0.45, 0.4969, 0.5096]
mean_lstmb_w2v = [0.47422, 0.45218, 0.47712, 0.4792333333, 0.47236]
plt.plot(epoch[0:len(mean_cnn_gnv)], mean_cnn_gnv, color='r', label = 'CNN Pre-trained')
plt.plot(epoch[0:len(mean_cnn_w2v)], mean_cnn_w2v, color='r', linestyle='dashed', label = 'CNN Post-trained')
plt.plot(epoch[0:len(mean_lstmu_gnv)], mean_lstmu_gnv, color='b', label = 'LSTM_Uni Pre-trained')
plt.plot(epoch[0:len(mean_lstmu_w2v)], mean_lstmu_w2v, color='b', linestyle='dashed', label = 'LSTM_Uni Post-trained')
plt.plot(epoch[0:len(mean_lstmb_gnv)], mean_lstmb_gnv, color='k', label = 'LSTM_Bi Pre-trained')
plt.plot(epoch[0:len(mean_lstmb_w2v)], mean_lstmb_w2v, color='k', linestyle='dashed', label = 'LSTM_Bi Post-trained')
plt.xlabel('Models', fontsize = font)
plt.ylabel('Validation Accuracy', fontsize = font)
plt.title('Validation Accurcay of Multi-classs Models', **title_font)
plt.show()

##########################################################################
# Non_binary Model Losses
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_cnn_gnv = [0.92399, 0.66784, 0.65001, 0.65123, 0.5930375, 0.5650142857, 0.621275, 0.606]
mean_cnn_w2v = [1.02797, 1.03056, 1.03521, 1.059077778, 1.053228571, 1.0162, 0.9681, 1.148]
mean_lstmu_gnv = [0.99231, 0.98749, 1.0026, 1.004414286, 1.066166667, 1.06]
mean_lstmu_w2v = [0.99608, 0.97508, 0.99616, 0.9665666667, 0.9786]
mean_lstmb_gnv = [1.00283, 0.99447, 0.99634, 0.9917444444, 0.9946333333, 0.9433]
mean_lstmb_w2v = [0.99519, 0.98591, 0.98858, 0.9813, 0.98676]
plt.plot(epoch[0:len(mean_cnn_gnv)], mean_cnn_gnv, color='r', label = 'CNN Pre-trained')
plt.plot(epoch[0:len(mean_cnn_w2v)], mean_cnn_w2v, color='r', linestyle='dashed', label = 'CNN Post-trained')
plt.plot(epoch[0:len(mean_lstmu_gnv)], mean_lstmu_gnv, color='b', label = 'LSTM_Uni Pre-trained')
plt.plot(epoch[0:len(mean_lstmu_w2v)], mean_lstmu_w2v, color='b', linestyle='dashed', label = 'LSTM_Uni Post-trained')
plt.plot(epoch[0:len(mean_lstmb_gnv)], mean_lstmb_gnv, color='k', label = 'LSTM_Bi Pre-trained')
plt.plot(epoch[0:len(mean_lstmb_w2v)], mean_lstmb_w2v, color='k', linestyle='dashed', label = 'LSTM_Bi Post-trained')
plt.xlabel('Models', fontsize = font)
plt.ylabel('Validation Losses', fontsize = font)
plt.title('Validation Losses of Multi-class Models', **title_font)
plt.show()
#########################################################################



