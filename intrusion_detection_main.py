# -*- coding: utf-8 -*-
"""

Dataset: NSL_KDD    - created by using the tcpdumps of the 1998 DARPA Intrusion Detection System (IDS) Evaluation dataset
                    - contains 41 features, training set contains 22 different attacks + normal traffic (According to orginal authors).
                    - contains 3 categorical features ('protocol_type', 'service', 'flag')
                    - Many papers view attacks as belonging to 5 classes: Normal, DoS, Probe, R2L and U2R (Not explored in current work)
                    
                    
Steps performed in code:    - Read in train and test sets (Most of this done in the utility file)
                            - Train XGBoost model to perform supervised intrusion detection
                            - Use TreeSHAP to explain model predictions (across training set and test set)
                            - Train autoencoder module using explanations from training set
                            - Perform anomaly detection on explanations from test set based on reconstruction error of the autonecoder
                            

"""



import os
os.chdir(' ')              	# location where files are stored (main + utility)
save_loc = ' '    			# location to save results
data_loc = ' '              # location of dataset

import numpy as np
import matplotlib.pyplot as plt
import pickle
import xgboost
import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import utility_funcs as utf

np.random.seed(10)
tf.random.set_seed(10)

""" *********************************** LOAD DATA (If code previously ran) ********************************** """
"""

data_kdd = pickle.load(open("{}/data_nsl.pkl".format(save_loc), "rb"))
results_model = pickle.load(open("{}/data_0.pkl".format(save_loc), "rb"))
model = pickle.load(open("{}/model.pkl".format(save_loc), "rb"))

results_AE_SHAP, history_AE_SHAP = pickle.load(open("{}/data_1.pkl".format(save_loc), "rb"))
autoencoder_shap = utf.Autoencoder(results_AE_SHAP['x_data'].shape[1], results_AE_SHAP['best_params'][2], results_AE_SHAP['best_params'][3]) # create the AE model object
autoencoder_shap.full.load_weights('{}/AE_shap_weights'.format(save_loc)); 
 
res_pca = pickle.load(open("{}/data_2.pkl".format(save_loc), "rb"))

"""
""" ******************************************************************************** """

# read in dataset, save train/test sets into data object, NB Y:Normal=0, Y:ATTACK=1
data_kdd = utf.read_KDD(data_loc) 

# train model using default settings of XGBoost 
model = xgboost.XGBClassifier(use_label_encoder=False, objective="binary:logistic", seed= 10) 
model.fit(data_kdd['X_train'], data_kdd['Y_train_bin'])

# compute performance statistics of the model, store in 'results_model' object
results_model = {}
results_model['y_pred_train'] = model.predict(data_kdd['X_train'])
results_model['y_pred_test'] = model.predict(data_kdd['X_test'])
results_model['y_pred_test_21'] = model.predict(data_kdd['X_test_21'])

results_model['performance_model_train'] = utf.compute_performance_stats(data_kdd['Y_train_bin'], results_model['y_pred_train'])
results_model['performance_model_test'] = utf.compute_performance_stats(data_kdd['Y_test_bin'], results_model['y_pred_test'])
results_model['performance_model_test_21'] = utf.compute_performance_stats(data_kdd['Y_test_bin_21'], results_model['y_pred_test_21'])

# SAVE DATA 
pickle.dump(data_kdd, open("{}/data_nsl.pkl".format(save_loc), "wb")) 
pickle.dump(model, open("{}/model.pkl".format(save_loc), "wb"))

# calculate how many new attacks the XGBoost model can identify, NB during deployment, we have no way of knowing these are new attacks
z = np.zeros(len(data_kdd['Y_test_bin']),)
z[data_kdd['new_attack_locs']] = 1
temp = np.multiply(z, results_model['y_pred_test'])
results_model['num_new_attacks_detected'] =   np.sum(temp)                                      # correctly identified 1297 new attacks (TP)
results_model['TPR_new_attacks'] = np.round(np.sum(temp)/len(data_kdd['new_attack_locs']) , 2)  # i.e., correctly identified 35% of new attacks (TPR)


""" ************************* Create AutoEncoder based on SHAP data **************************** """
# compute SHAP values of model across train and test sets
results_AE_SHAP = {}
explainer = shap.TreeExplainer(model, data_kdd['X_train'], feature_perturbation = "interventional", model_output='probability') # NB output='probability' decomposes inputs among Pr(Y=1='Attack'|X)
results_AE_SHAP['shap_train'] = explainer.shap_values(data_kdd['X_train'])
results_AE_SHAP['shap_test'] = explainer.shap_values(data_kdd['X_test'])

# Compute example explanation for a Probe attack - NB SHAP package requires us to wrap everything into a single object before plotting
idx = data_kdd['Y_train'][data_kdd['Y_train']=='ipsweep'].index # get all locations where ipsweep attack occur (i.e., main attack class = Probe)

class Object(object):
    pass

exp = Object
exp.feature_names = data_kdd['feature_names']
exp.base_values = explainer.expected_value
exp.data = data_kdd['X_train'].loc[idx[0]]
exp.values = results_AE_SHAP['shap_train'][idx[0]]

shap.plots.waterfall(exp) # plot the explanation

# scale data using Sklearn minmax scaler
results_AE_SHAP['scaler'] = MinMaxScaler(feature_range=(-1,1))                                     
results_AE_SHAP['shap_train_scaled'] = results_AE_SHAP['scaler'].fit_transform(results_AE_SHAP['shap_train'])  # scale the training set data
results_AE_SHAP['shap_test_scaled'] = results_AE_SHAP['scaler'].transform(results_AE_SHAP['shap_test'])        # scale the test set data

# before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
results_AE_SHAP['x_data'], results_AE_SHAP['val_data'] = train_test_split(results_AE_SHAP['shap_train_scaled'], test_size=0.2, random_state=10)


# perform grid search to find the best paramters to use for the autoencoder model
# specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
results_AE_SHAP['parameters'] = {
                                    # encoder params to search across
                                    'dense_1_units':[1456],                                   'dense_1_activation':['relu'],  # 32,64,128,256
                                    'dense_2_units':[724],                                 'dense_2_activation':['relu'],  # 32,48,64
                                    'dense_3_units':[14],                                           'dense_3_activation':['relu'], # 8,16, 24
                                    # decoder params to search across
                                    'dense_4_units':[632],                                        'dense_4_activation':['relu'],
                                    'dense_5_units':[1644],                                    'dense_5_activation':['relu'],  # 64,128,256
                                    'dense_6_units':[results_AE_SHAP['x_data'].shape[1]],               'dense_6_activation':['tanh']    
                                }

# check size of grid space to ensure not too large
z = [*results_AE_SHAP['parameters'].values()]                       # get values of each sublist in the overall parameter list 
z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 

# perform the grid search and return parameters of the best model
results_AE_SHAP['grid_search'], results_AE_SHAP['best_params'] = utf.get_hyper_Autoencoder(results_AE_SHAP['parameters'], results_AE_SHAP['x_data'], results_AE_SHAP['val_data'],
                                                                                           method='exact', num_epochs=10, batch_size=512, AE_type = 'joint')   
# Using the best parameters, build and train the final model
autoencoder_shap = utf.Autoencoder(results_AE_SHAP['x_data'].shape[1], results_AE_SHAP['best_params'][2], results_AE_SHAP['best_params'][3], AE_type='joint') # create the AE model object

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
history_AE_SHAP = autoencoder_shap.full.fit(results_AE_SHAP['x_data'], results_AE_SHAP['x_data'], epochs=1000, batch_size=512, shuffle=True, validation_data=(results_AE_SHAP['val_data'],   
                               results_AE_SHAP['val_data']), verbose=2, callbacks=[early_stop]).history
# plot the training curve
plt.plot(history_AE_SHAP["loss"], label="Training Loss")
plt.plot(history_AE_SHAP["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


# perform anomaly detection based on the reconstruction error of the AE and save results
results_AE_SHAP['performance_new_attacks'], results_AE_SHAP['new_attack_pred_locs'], results_AE_SHAP['AE_threshold'] = utf.AE_anomaly_detection(autoencoder_shap, results_AE_SHAP['shap_train_scaled'], 
                                                                                                                                                results_AE_SHAP['shap_test_scaled'], data_kdd['new_attack_locs'], plt_title='Autoencoder trained on SHAP values')

# calculate overall accuracy of the IDS system (XGBoost IDS and Anomaly detector) to detect attacks new or old attacks on the NSL-KDD Testset+
results_AE_SHAP['all_attack_pred_locs'] = np.unique(np.concatenate((results_AE_SHAP['new_attack_pred_locs'], np.where(results_model['y_pred_test']==1)[0] )))
results_AE_SHAP['y_pred_all'] = np.zeros(len(data_kdd['Y_test_bin']),)
results_AE_SHAP['y_pred_all'][results_AE_SHAP['all_attack_pred_locs']] = 1
results_AE_SHAP['performance_overall'] = utf.compute_performance_stats(data_kdd['Y_test_bin'], results_AE_SHAP['y_pred_all'])


# calculate TPR of new attacks (exclusively) for the autonencoder as well the overall NIDS
results_model['TPR_new_attacks'] = np.round(results_AE_SHAP['performance_new_attacks']['TP']/len(data_kdd['new_attack_locs']) , 2)

results_model['TPR_new_attacks_overall'] = np.round(results_AE_SHAP['performance_overall']['TP']/np.sum(data_kdd['Y_test_bin']) , 2)

# SAVE DATA
pickle.dump(results_model, open("{}/data_0.pkl".format(save_loc), "wb")) 
pickle.dump([results_AE_SHAP, history_AE_SHAP], open("{}/data_1.pkl".format(save_loc), "wb")) 
autoencoder_shap.full.save_weights('{}AE_shap_weights'.format(save_loc))
   
# z = autoencoder_shap.full.predict(results_AE_SHAP['shap_test_scaled'])
# z.min()
# z.max()





""" ************************* Analysis of results **************************** """

"""  
    To try and understand why anomaly-detection based on SHAP (i.e. the 'explanation domain') results in enhanced performance of the NIDS,
    we visualise the data using PCA: First, we apply PCA to the SHAP explanations computed on the
    training data, and visualise the class clusters. Then we apply PCA to the raw training data, and visualise the class clusters.
    
    The final results show that SHAP values produce strong separation between normal and attacks in PCA space, while there is still much overlap
    between classes in the raw input space.
    
    SHAPs ability to distinguish the classes like this may be based on the fact that SHAP is a linear model, ie the complex behaviour of the model
    is being explained in a linear fashion -> shap transforms non-linear data into linear space. Moreover, because SHAP decomposes each the feature's 
    importance into the model output, in this case probaility space, the PCA results tend to fall within a small linear range, not requiring any normalisation
    as in the case of the raw inputs.   
    
    In addition, we see that only 32% of the data variance is explained by the first 2 components of PCA_RAW, while for PCA-SHAP, the first 2 components
    are able to account for 88% of the variance. This suggests that SHAP is able to compress a lot of important information efficiently.
    
"""

res_pca = {} # create object to store results of our PCA analysis

# first, transform SHAP data from training set into PCA space
res_pca['model_SHAP'] = PCA(n_components=10)                                   # create PCA object to represent the SHAP data
res_pca['SHAP_train'] = res_pca['model_SHAP'].fit_transform(results_AE_SHAP['shap_train'])    # fit PCA to the SHAP training data
# res_pca.model_SHAP.explained_variance_ratio_.cumsum()


# now plot the first 2 PCA components fitted to the SHAP training data
fig = plt.figure(figsize=(8,8))
plt.scatter(res_pca['SHAP_train'][data_kdd['train_attack_locs'],0], res_pca['SHAP_train'][data_kdd['train_attack_locs'],1], marker='1', s=1, c='crimson', label='Attacks')   # plot the attacks in red
plt.scatter(res_pca['SHAP_train'][data_kdd['train_normal_locs'],0], res_pca['SHAP_train'][data_kdd['train_normal_locs'],1], marker='4', s=1, c='royalblue', label='Normal')   # plot the normal traffic in blue
# plt.title('PCA Applied to SHAP values of the NSL-KDD Training Dataset')
plt.xlabel('Component 1', fontsize=15)
plt.ylabel('Component 2', fontsize=15)

lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=15)
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]

plt.show()


# NB need to scale inputs using only standard deviation, NB this produces results similar to those seen in many other intrusion works

#(np.count_nonzero(data_kdd.X_train==0) > ((len(data_kdd.X_train)*41)/2) )              # NSL Dataset is sparse, so better not to deduct mean when standardising the data
data_kdd['scaler'] = StandardScaler(with_mean=False)                                    # scale the training set data
data_kdd['X_train_scaled'] = data_kdd['scaler'].fit_transform(data_kdd['X_train'])


res_pca['model_x'] = PCA(n_components=10)                                               # create PCA object to represent the raw input data
res_pca['x_train'] = res_pca['model_x'].fit_transform(data_kdd['X_train_scaled'])       # fit PCA to the raw training data
# res_pca.model_x.explained_variance_ratio_.cumsum()


fig = plt.figure(figsize=(8,8))   # default size = 8,6
plt.scatter(res_pca['x_train'][data_kdd['train_attack_locs'],0], res_pca['x_train'][data_kdd['train_attack_locs'],1], marker='1', s=1, c='crimson', label='Attacks')   # plot the attacks in red
plt.scatter(res_pca['x_train'][data_kdd['train_normal_locs'],0], res_pca['x_train'][data_kdd['train_normal_locs'],1], marker='4', s=1, c='royalblue', label='Normal')   # plot the normal traffic in blue
# plt.title('PCA Applied to the raw inputs of the NSL-KDD Training Dataset')
plt.xlabel('Component 1', fontsize=15)
plt.ylabel('Component 2', fontsize=15)

lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=15)
lgnd.legendHandles[0]._sizes = [100]
lgnd.legendHandles[1]._sizes = [100]

plt.show()

# plot the amount of variance explained by each component for the case of SHAP and Raw inputs
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
label_locs = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(label_locs - width/2, res_pca['model_SHAP'].explained_variance_ratio_.cumsum(), width, label='PCA SHAP')
rects2 = ax.bar(label_locs + width/2, res_pca['model_x'].explained_variance_ratio_.cumsum(), width, label='PCA Raw Input')
ax.set_ylabel('Fraction of Explained Variance')
ax.set_xlabel('PCA Components')
ax.set_title('Cumulative Fraction of Variance explained by each PCA Component')
ax.set_xticks(label_locs)
ax.set_xticklabels(labels)
ax.legend()

del labels, rects1, rects2, width, label_locs

# SAVE DATA
pickle.dump(data_kdd, open("{}/data_nsl.pkl".format(save_loc), "wb")) 
pickle.dump(res_pca, open("{}/data_2.pkl".format(save_loc), "wb")) 









#EoF
