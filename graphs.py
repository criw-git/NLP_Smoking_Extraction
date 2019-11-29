import matplotlib.pyplot as plt
import numpy as np

N = 6 # change to 6 and add LSTMU and LSTMB Models
font = 'x-large'
title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'}
width = 0.10
# ###########################################################################
# # Binary DL Graph Models
# f1_means = (0.853993887, 0.811600324, 0.8017485516, 0.8069866652, 0.8024736733, 0.8004409841)
# f1_err = (0.0161547238553041, 0.00414287866088785, 0.02670095987, 0.01426782532, 0.03579518095, 0.01343905201)
# precision_means = (0.888551695, 0.68561981031667, 0.6881050722, 0.6924379413, 0.6919641535, 0.688190854)
# precision_err = (0.0418452561710614, 0.00116344084501155, 0.006112392218, 0.005777803157, 0.01871322656, 0.004446866707)
# recall_means = (0.824293785310734, 0.994350282485875, 0.9638418064, 0.9683615819, 0.9661016949, 0.9574011299)
# recall_err = (0.0337357474989223, 0.0106532095094021, 0.07314147954, 0.0448195097, 0.1071958529, 0.03856936352)
# accuracy_means = (0.806589147286821, 0.683333333333333, 0.6751937949, 0.6829457364, 0.6786821705, 0.6731007752)
# accuracy_err = (0.0232880912958136, 0.00518408337150901, 0.02550166437, 0.01389114218, 0.0232880913, 0.01459456596)

# fig, ax = plt.subplots()

# ind = np.arange(N) 

# ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
# ax.bar(ind + width, precision_means, width,
#     label='Precision', yerr=precision_err)
# ax.bar(ind + 2*width, recall_means, width,
#     label='Recall', yerr=recall_err)
# ax.bar(ind + 3*width, accuracy_means, width,
#     label='Accuracy', yerr=accuracy_err)

# x_labels = ['CNN Pre', 'CNN Post', 'LSTMU Pre', 'LSTMU Post', 'LSTMB Pre', 'LSTMB Post']
# ax.set_ylabel('Performance', fontsize = font)
# ax.set_xticks(ind + width + width/2)
# ax.set_xticklabels(x_labels) 
# ax.set_xlabel('Models', fontsize = font)
# ax.set_title('Binary DL Model Performances', **title_font)
# ax.tick_params(axis='both', which='major', labelsize=16)
# plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

# plt.show()
# ##################################################################################
# # Non-Binary DL Bar Graph
# f1_means = (0.680436811, 0.397195236, 0.4134755668, 0.422475575648325, 0.5329457364, 0.501)
# precision_means = (0.711238296411343, 0.40775712823729, 0.4081667731, 0.387854684047091, 0.5329457364, 0.501)
# recall_means = (0.693798449612402, 0.512790697674418, 0.5313953488, 0.51860465116279, 0.5329457364, 0.501)
# accuracy_means = (0.693798449612403, 0.512790697674418, 0.5313953488, 0.51860465116279, 0.5329457364, 0.501)
# f1_err = (0.0399069890331443, 0.072036577433337, 0.02701856799, 0.0270013828625037, 0.02334536305, 0.05159538942)
# precision_err = (0.0380529926709191, 0.0751936234383296, 0.08495896048, 0.0623189223876878, 0.02334536305, 0.05159538942)
# recall_err = (0.0305194103643869, 0.0654734166653383, 0.01689001638, 0.0368069692621588, 0.02334536305, 0.05159538942)
# accuracy_err = (0.0305194103643868, 0.0654734166653383, 0.01689001638, 0.0368069692621588, 0.02334536305, 0.05159538942)

# fig, ax = plt.subplots()

# ind = np.arange(N) 

# ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
# ax.bar(ind + width, precision_means, width,
#     label='Precision', yerr=precision_err)
# ax.bar(ind + 2*width, recall_means, width,
#     label='Recall', yerr=recall_err)
# ax.bar(ind + 3*width, accuracy_means, width,
#     label='Accuracy', yerr=accuracy_err)

# x_labels = ['CNN Pre', 'CNN Post', 'LSTMU Pre', 'LSTMU Post', 'LSTMB Pre', 'LSTMB Post']
# ax.set_ylabel('Performance', fontsize = font)
# ax.set_xticks(ind + width + width/2)
# ax.set_xticklabels(x_labels) 
# ax.set_xlabel('Models', fontsize = font)
# ax.set_title('Multi-class DL Model Performances', **title_font)
# ax.tick_params(axis='both', which='major', labelsize=16)
# plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

# plt.show()
# ########################################################################
# N = 3
# # Binary CL Graph Models
# f1_means = (0.785882352941176, 0.723032069970845, 0.725146198830409)
# f1_err = (0, 0, 0)
# precision_means = (1, 0.74251497005988, 0.74251497005988)
# precision_err = (0, 0, 0)
# recall_means = (0.647286821705426, 0.704545454545454, 0.708571428571428)
# recall_err = (0, 0, 0)
# accuracy_means = (0.647286821705426, 0.631782945736434, 0.635658914728682)
# accuracy_err = (0, 0, 0)

# fig, ax = plt.subplots()

# ind = np.arange(N)

# ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
# ax.bar(ind + width, precision_means, width,
#     label='Precision', yerr=precision_err)
# ax.bar(ind + 2*width, recall_means, width,
#     label='Recall', yerr=recall_err)
# ax.bar(ind + 3*width, accuracy_means, width,
#     label='Accuracy', yerr=accuracy_err)

# x_labels = ['NB', 'SVM', 'LR']
# ax.set_ylabel('Performance', fontsize = font)
# ax.set_xticks(ind + width + width/2)
# ax.set_xticklabels(x_labels) 
# ax.set_xlabel('Models', fontsize = font)
# ax.set_title('Binary Classical Model Performances', **title_font)
# plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
# ax.tick_params(axis='both', which='major', labelsize=16)

# plt.show()
# ##################################################################################
# # Non-Binary CL Graph Models
# f1_means = (0.696837964690222, 0.614022779305171, 0.56230920910151)
# f1_err = (0, 0, 0)
# precision_means = (0.971012793191342, 0.690293035849319, 0.592131254365122)
# precision_err = (0, 0, 0)
# recall_means = (0.550387596899224, 0.565891472868217, 0.542635658914728)
# recall_err = (0, 0, 0)
# accuracy_means = (0.550387596899224, 0.565891472868217, 0.542635658914728)
# accuracy_err = (0, 0, 0)

# fig, ax = plt.subplots()

# ind = np.arange(N) 

# ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
# ax.bar(ind + width, precision_means, width,
#     label='Precision', yerr=precision_err)
# ax.bar(ind + 2*width, recall_means, width,
#     label='Recall', yerr=recall_err)
# ax.bar(ind + 3*width, accuracy_means, width,
#     label='Accuracy', yerr=accuracy_err)

# x_labels = ['NB', 'SVM', 'LR']
# ax.set_ylabel('Performance', fontsize = font)
# ax.set_xticks(ind + width + width/2)
# ax.set_xticklabels(x_labels) 
# ax.set_xlabel('Models', fontsize = font)
# ax.set_title('Multi-class Classical Model Performances', **title_font)
# plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
# ax.tick_params(axis='both', which='major', labelsize=16)

# plt.show()
##################################################################################
N = 4
# Binary CL Graph Models
model_1_metrics = (0.785882352941176, 1, 0.647286821705426, 0.647286821705426)
model_1_err = (0, 0, 0, 0)
model_2_metrics = (0.723032069970845, 0.74251497005988, 0.704545454545454, 0.631782945736434)
model_2_err = (0, 0, 0, 0)
model_3_metrics = (0.725146198830409, 0.74251497005988, 0.708571428571428, 0.635658914728682)
model_3_err = (0, 0, 0, 0)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N)

ax.bar(ind, model_1_metrics, width, label='Naive Bayes', yerr=model_1_err)
ax.bar(ind + width, model_2_metrics, width,
    label='SVM', yerr=model_2_err)
ax.bar(ind + 2*width, model_3_metrics, width,
    label='Logistic Regression', yerr=model_3_err)

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Binary Classical Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()
#######################################################################
N = 4
# Non-Binary CL Graph Models 2
model_1_metrics = (0.696837964690222, 0.971012793191342, 0.550387596899224, 0.550387596899224)
model_1_err = (0, 0, 0, 0)
model_2_metrics = (0.614022779305171, 0.690293035849319, 0.565891472868217, 0.565891472868217)
model_2_err = (0, 0, 0, 0)
model_3_metrics = (0.56230920910151, 0.592131254365122, 0.542635658914728, 0.542635658914728)
model_3_err = (0, 0, 0, 0)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N) 

ax.bar(ind, model_1_metrics, width, label='Naive Bayes', yerr=model_1_err)
ax.bar(ind + width, model_2_metrics, width,
    label='SVM', yerr=model_2_err)
ax.bar(ind + 2*width, model_3_metrics, width,
    label='Logistic Regression', yerr=model_3_err)

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Multi-class Classical Model Performances', **title_font)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.show()
##########################################################################333
# Binary DL Graph Models 2
model_1_metrics = (0.853993887, 0.888551695, 0.824293785310734, 0.806589147286821)
model_1_err = (0.0161547238553041, 0.0418452561710614, 0.0337357474989223, 0.0232880912958136)
model_2_metrics = (0.811600324, 0.68561981031667,  0.994350282485875, 0.683333333333333)
model_2_err = (0.00414287866088785, 0.00116344084501155, 0.0106532095094021, 0.005184083371509010)
model_3_metrics = (0.8017485516, 0.6881050722, 0.9638418064, 0.6751937949)
model_3_err = (0.02670095987, 0.006112392218, 0.07314147954, 0.02550166437)
model_4_metrics = (0.8069866652, 0.6924379413, 0.9683615819, 0.6829457364)
model_4_err = (0.01426782532, 0.005777803157, 0.0448195097, 0.01389114218)
model_5_metrics = (0.8024736733, 0.6919641535, 0.96610169497, 0.6786821705)
model_5_err = (0.03579518095, 0.01871322656, 0.1071958529, 0.0232880913)
model_6_metrics = (0.8004409841, 0.688190854, 0.9574011299, 0.6731007752)
model_6_err = (0.01343905201, 0.004446866707, 0.03856936352, 0.01459456596)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N) 

ax.bar(ind, model_1_metrics, width, label='CNN Pre', yerr=model_1_err)
ax.bar(ind + width, model_2_metrics, width,
    label='CNN Post', yerr=model_2_err)
ax.bar(ind + 2*width, model_3_metrics, width,
    label='LSTMU Pre', yerr=model_3_err)
ax.bar(ind + 3*width, model_4_metrics, width, label='LSTMU Post', yerr=model_4_err)
ax.bar(ind + 4*width, model_5_metrics, width,
    label='LSTMB Pre', yerr=model_5_err)
ax.bar(ind + 5*width, model_6_metrics, width,
    label='LSTMP Post', yerr=model_6_err)

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Binary DL Model Performances', **title_font)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.show()
###################################################################################
# Non-Binary DL Bar Graph
model_1_metrics = (0.680436811, 0.711238296411343, 0.693798449612402, 0.693798449612403)
model_1_err = (0.0399069890331443, 0.0380529926709191, 0.0305194103643869, 0.0305194103643868)
model_2_metrics = (0.397195236, 0.40775712823729,  0.512790697674418, 0.512790697674418)
model_2_err = (0.072036577433337, 0.0751936234383296, 0.0654734166653383, 0.0654734166653383)
model_3_metrics = (0.4134755668, 0.4081667731, 0.5313953488, 0.5313953488)
model_3_err = (0.0270185679, 0.08495896048, 0.01689001638, 0.01689001638)
model_4_metrics = (0.422475575648325, 0.387854684047091, 0.51860465116279, 0.51860465116279)
model_4_err = (0.0270013828625037, 0.0623189223876878, 0.0368069692621588, 0.0368069692621588)
model_5_metrics = (0.5329457364, 0.53294573645, 0.5329457364, 0.5329457364)
model_5_err = (0.02334536305, 0.02334536305, 0.02334536305, 0.02334536305)
model_6_metrics = (0.501, 0.501, 0.501, 0.501)
model_6_err = (0.05159538942, 0.05159538942, 0.05159538942, 0.05159538942)

fig, ax = plt.subplots(figsize=(15, 10))

ind = np.arange(N) 

ax.bar(ind, model_1_metrics, width, label='CNN Pre', yerr=model_1_err)
ax.bar(ind + width, model_2_metrics, width,
    label='CNN Post', yerr=model_2_err)
ax.bar(ind + 2*width, model_3_metrics, width,
    label='LSTMU Pre', yerr=model_3_err)
ax.bar(ind + 3*width, model_4_metrics, width, label='LSTMU Post', yerr=model_4_err)
ax.bar(ind + 4*width, model_5_metrics, width,
    label='LSTMB Pre', yerr=model_5_err)
ax.bar(ind + 5*width, model_6_metrics, width,
    label='LSTMP Post', yerr=model_6_err)

x_labels = ['F1', 'Precision', 'Recall', 'Accuracy']
ax.set_ylabel('Performance', fontsize = font)
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels) 
ax.set_xlabel('Metrics', fontsize = font)
ax.set_title('Multi-class DL Model Performances', **title_font)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.show()
