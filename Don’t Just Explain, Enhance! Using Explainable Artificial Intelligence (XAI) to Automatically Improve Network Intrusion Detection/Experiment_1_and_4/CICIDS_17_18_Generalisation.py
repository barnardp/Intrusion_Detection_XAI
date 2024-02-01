
"""
    Code that performs the inter and intra valuation procedure on the CICIDS17/18 datasets as described from: 
        "Towards Model Generalization for Intrusion Detection: Unsupervised Machine Learning Techniques"
    
"""

import os
os.chdir('D:/')                                 # location where utility file is located
save_loc = 'D:/'                                # location to save results
data_loc_17 = 'D:/Datasets/CICIDS2017/original/GeneratedLabelledFlows/TrafficLabelling/'            # location of CICIDS17 dataset
new_data_loc_17 ='D:/Datasets/CICIDS2017/original/clean/'
data_loc_18 = 'D:/Datasets/CICIDS2018/Processed Traffic Data for ML Algorithms/'                    # location of CICIDS18 dataset
new_data_loc_18 ='D:/Datasets/CICIDS2018/clean/'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost
from sklearn import metrics #precision_recall_curve, roc_curve, auc
import shap
import  utility_funcs as utf
from sklearn.preprocessing import  MinMaxScaler
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
from os import listdir
from os.path import isfile, join
import gc
from scipy.special import expit
import itertools
from tqdm import tqdm

   

"""  RUN Pre-processing Steps for CICIDS17/18 Datatsets. NB  can skip this step if it has previously been run """
"""
utf.clean_dataset(data_loc=data_loc_17, new_data_loc=new_data_loc_17)               # create 'clean' copy of the CICIDS17 flow datasets
utf.aggregate_data(DATA_DIR=new_data_loc_17)                                        # create separate files for the malicious and benign data, sub-sample benign to 1M samples
utf.generate_clean_data(data_loc=data_loc_18, new_data_loc=new_data_loc_18)         # create 'clean' copy of the CICIDS18 flow datasets
utf.GroundTruthGeneration(DATA_DIR=new_data_loc_18)                                 # create separate files for the malicious and benign data, sub-sample benign to 1M samples
"""
""" End of Pre-processing steps """

# Load CICIDS17 Data. Note We assume there is a typo in Fig 1 of the original paper. Here, they say that there are 556, 556 samples of malicious data
# Howwever, after careful investigation, we find that this is only the case when the CICIDS17 dataset is filtered for NANs and inf values. Once duplicates
# are removed, the actual figure is reduced to 425,741. This implies that only there would only be 127,722 attacks in the validation set, ie 30% of 1,341,787.
data_17 = utf.load_data_cicids_miels(new_data_loc_17, "all_benign.csv", train_size=50000, test_size=200000, verbose=False, validation_perc=0.3)
data_18 = utf.load_data_cicids_miels(new_data_loc_18, "benign_1M.csv", train_size=50000, test_size=500000, verbose=False, validation_perc=0.15) # Load CICIDS18 Data
# Check everything lines up with Fig 1 of original paper
"""
data_17['y_train'].value_counts()           # 50k benign=0 labels
data_17['y_validation'].value_counts()      # 60k benign=0, 127,722 Attack=1
data_17['y_test'].value_counts()            # 140k benign=0, 298019 Attack=1
data_17['label_train'].value_counts()       # 50k benign=0 labels
data_17['label_validation'].value_counts()  # 60k Benign, rest all attacks
data_17['label_test'].value_counts()        # 140k benign, rest all attacks
data_18['y_train'].value_counts()           # 50k benign=0 labels
data_18['y_validation'].value_counts()      # 75k benign=0, 201,268 Attack=1
data_18['y_test'].value_counts()            # 425k benign=0, 1,140,519 Attack=1
"""

""" 
# LOAD Previosuly SAVED Data
model = pickle.load(open(save_loc + "model.pkl", "rb"))
results_17_18 = pickle.load(open(save_loc + "results_17_18_shap.pkl", "rb"))
results_percentile_17_18 = pickle.load(open(save_loc + "results_percentile_17_18.pkl", "rb"))

"""

""" ************************** ANALYSE RESULTS FOR CICIDS2017 AS PRIMARY DATASET ************************** """

# change some of the XGBoost default paramters to help avoid over fitting
parameters = {                                # define the grid search parameters
                "max_depth": 6,               # max depth of each decision tree, typically 1-10, default=6
                "eta": 0.01,                  # learning rate typically 0.01 - 0.2, max 1.0, default = 0.3, Lower values avoid over-fitting.      
                "subsample": 0.75,            # Fract[training set] to train each tree, if too low->underfit, too high->overfit, default 1.0, Lower ratios avoid over-fitting.
                "colsample_bytree": 0.75      # fract[features] to train each tree, default = 1, Lower ratios avoid over-fitting.
            }

# train model on CICIDS 17 Dataset, using default settings of XGBoost. NB here, we combine train set with val set so that we can train a supervised model
temp1 = pd.concat([data_17['x_train'], data_17['x_validation']], ignore_index=True)    
temp2 = pd.concat([data_17['y_train'], data_17['y_validation']], ignore_index=True) #  temp2.value_counts() ->110,000 class 0, 127,722 class 1

model = xgboost.XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric='logloss', seed= 10,  **parameters) 
model.fit(temp1, temp2)    

results_17_18 = {}
# compute intra performance statistics of the model trained and tested on CICIDS17 
y_pred = model.predict(data_17['x_test']) 
y_prob = model.predict_proba(data_17['x_test'])[:,1]
results_17_18['results_model_intra'] = utf.compute_performance_stats(data_17['y_test'], y_pred, y_prob)
# compute inter performance statistics of the model trained and on CICIDS17 and tested CICIDS18
y_pred = model.predict(data_18['x_test']) 
y_prob = model.predict_proba(data_18['x_test'])[:,1]
results_17_18['results_model_inter'] = utf.compute_performance_stats(data_18['y_test'], y_pred, y_prob)
  
# compute SHAP values of model across train and test sets
explainer = shap.TreeExplainer(model, data_17['x_train'], feature_perturbation = "interventional", model_output='probability') # NB output='probability' decomposes inputs among Pr(Y=1='Attack'|X)
dt = time.time()
results_17_18['shap_17_train'] = explainer.shap_values(data_17['x_train'])
dt = (time.time() - dt)/len( data_17['x_train'] )
results_17_18['ave_time_exp'] = dt 
results_17_18['shap_17_val'] = explainer.shap_values(data_17['x_validation'])
results_17_18['shap_17_test'] = explainer.shap_values(data_17['x_test'])
results_17_18['shap_18_test'] = explainer.shap_values(data_18['x_test'])   # takes 83 mins to do 1,565,519 explanations
# results_17_18['shap_18_train'] = explainer.shap_values(data_18['x_train'])
# results_17_18['shap_18_val'] = explainer.shap_values(data_18['x_validation'])

# SAVE DATA 
pickle.dump(results_17_18, open(save_loc + "results_17_18_shap.pkl", "wb")) 
pickle.dump(model, open(save_loc + "model.pkl", "wb"))

""" ************************** Train Autoencoder on CICIDS2017 ************************** """

# scale data using Sklearn minmax scaler, NB here we fit the scaler to both the train and val normal data
scaler = MinMaxScaler(feature_range=(-1,1))          
scaler.fit( np.concatenate((results_17_18['shap_17_train'], results_17_18['shap_17_val'][ np.where( data_17['label_validation'] == "Benign" )[0] ]), axis=0) );                         
shap_train_scaled = scaler.transform(results_17_18['shap_17_train'])   # scale the training set data
shap_val_scaled = scaler.transform(results_17_18['shap_17_val'])           # scale the validation set data
shap_test_A_scaled = scaler.transform(results_17_18['shap_17_test'])         # scale the test set data
shap_test_B_scaled = scaler.transform(results_17_18['shap_18_test'])         # scale the test set data

# before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
#x_data, val_data = train_test_split(shap_train_scaled, test_size=0.2, random_state=10)

x_data = shap_train_scaled.copy()
val_data = shap_val_scaled[ np.where( data_17['label_validation'] == "Benign" )[0] ]

# perform grid search to find the best paramters to use for the autoencoder model
# specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
parameters = {
                                    # encoder params to search across
                                    'dense_1_units':[1048],                                   'dense_1_activation':['relu'],  # 32,64,128,256
                                    'dense_2_units':[172],                                 'dense_2_activation':['relu'],  # 32,48,64
                                    'dense_3_units':[18],                                           'dense_3_activation':['relu'], # 8,16, 24
                                    # decoder params to search across
                                    'dense_4_units':[816],                                        'dense_4_activation':['relu'],
                                    'dense_5_units':[1480],                                    'dense_5_activation':['relu'],  # 64,128,256
                                    'dense_6_units':[x_data.shape[1]],               'dense_6_activation':['tanh']    
                                }

# check size of grid space to ensure not too large
z = [*parameters.values()]                       # get values of each sublist in the overall parameter list 
z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 

# perform the grid search and return parameters of the best model
_, results_17_18['best_params'] = utf.get_hyper_Autoencoder(parameters, x_data, val_data, method='exact', num_epochs=10, batch_size=1024, AE_type = 'joint', loss_monitor='val')   

# results_17_18['best_params'] = [[], [], [1048, 172, 18, 816, 1480, 67], ['relu','relu','relu','relu','relu','tanh']] # Model params for normal val, xgboost depth = 6

# Using the best parameters, build and train the final model
AE_model_17 = utf.Autoencoder(x_data.shape[1], results_17_18['best_params'][2], results_17_18['best_params'][3], AE_type='joint', learning_rate=1e-3) # create the AE model object

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=25, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
history_AE_SHAP = AE_model_17.full.fit(x_data, x_data, epochs=1000, batch_size=1024, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
# plot the training curve
plt.plot(history_AE_SHAP["loss"], label="Training Loss")
plt.plot(history_AE_SHAP["val_loss"], label="Validation Loss")
plt.title("Training curve for Autoencoder trained on CICIDS2017")
plt.xlabel("No. of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# SAVE DATA 
AE_model_17.full.save_weights(save_loc + 'AE_model_17_weights')
                           
                           
ground_truth_locs = np.where(data_17['y_test'] == 1)[0]
# use the explanations to identify the new attack samples on the CICIDS17 Dataset (Intra)
temp, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model_17, x_data, shap_test_A_scaled, ground_truth_locs, plt_title='Autoencoder trained on SHAP values from CICIDS2017', threshold=95, new_attack_stats=False)
# get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
y_prob = model.predict_proba(data_17['x_test'])[:, 1]
y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
y_pred = (y_prob>0.5)*1
# compute final results
results_17_18['results_overall_intra'] = utf.compute_performance_stats(data_17['y_test'], y_pred, y_prob)


ground_truth_locs = np.where(data_18['y_test'] == 1)[0]
# use the explanations to identify the new attack samples on the CICIDS17 Dataset (Intra)
temp, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model_17, x_data, shap_test_B_scaled, ground_truth_locs, plt_title='Autoencoder trained on CICIDS2017 Applied to CICIDS2018', threshold=95, new_attack_stats=False)
# get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
y_prob = model.predict_proba(data_18['x_test'])[:, 1]
y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
y_pred = (y_prob>0.5)*1
# compute final results
results_17_18['results_overall_inter'] = utf.compute_performance_stats(data_18['y_test'], y_pred, y_prob)

# SAVE DATA 
pickle.dump(results_17_18, open(save_loc + "results_17_18_final.pkl", "wb"))

results_percentile = pd.DataFrame()

for i in range(85, 99):
    
    _, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model_17, x_data, shap_test_B_scaled, ground_truth_locs, plt_title='Autoencoder - CICIDS2017 -> CICIDS2018 - Perc('+str(i)+')', threshold=i, new_attack_stats=False)
    # get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
    y_prob = model.predict_proba(data_18['x_test'])[:, 1]
    y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
    y_pred = (y_prob>0.5)*1
    # compute final results
    temp = utf.compute_performance_stats(data_18['y_test'], y_pred, y_prob)
    temp['Threshold'] = i
    results_percentile = results_percentile.append( temp, ignore_index=True)

# SAVE DATA 
pickle.dump(results_percentile, open(save_loc + "results_percentile_17_18.pkl", "wb"))






""" ************************** ANALYSE RESULTS FOR CICIDS2018 DATASET ************************** """
# NB running this half of the code concurrently with the first half might lead to memory issues. If this occurs, the code may have to be run from scratch
# starting from here (but remember to run the load the data_17/18 datasets first)
""" 
# LOAD Previosuly SAVED Data
model = pickle.load(open(save_loc + "model_18.pkl", "rb"))
results_18_17 = pickle.load(open(save_loc + "results_18_17_final.pkl", "rb"))
results_percentile_18 = pickle.load(open(save_loc + "results_percentile_18.pkl", "rb"))

"""

# change some of the XGBoost default paramters to help avoid over fitting
parameters = {                                # define the grid search parameters
                "max_depth": 6,               # max depth of each decision tree, typically 1-10, default=6
                "eta": 0.01,                  # learning rate typically 0.01 - 0.2, max 1.0, default = 0.3, Lower values avoid over-fitting.      
                "subsample": 1.0,            # Fract[training set] to train each tree, if too low->underfit, too high->overfit, default 1.0, Lower ratios avoid over-fitting.
                "colsample_bytree": 0.75      # fract[features] to train each tree, default = 1, Lower ratios avoid over-fitting.
            }

# train model on CICIDS 18 Dataset, using default settings of XGBoost. NB here, we combine train set with val set so that we can train a supervised model
temp1 = pd.concat([data_18['x_train'], data_18['x_validation']], ignore_index=True)    
temp2 = pd.concat([data_18['y_train'], data_18['y_validation']], ignore_index=True) #  temp2.value_counts()

model = xgboost.XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric='logloss', seed= 10,  **parameters) 
model.fit(temp1, temp2)    

results_18_17 = {}
# compute intra performance statistics of the model trained and tested on CICIDS17 
y_pred = model.predict(data_18['x_test']) 
y_prob = model.predict_proba(data_18['x_test'])[:,1]
results_18_17['results_model_intra'] = utf.compute_performance_stats(data_18['y_test'], y_pred, y_prob)
# compute inter performance statistics of the model trained and on CICIDS17 and tested CICIDS18
y_pred = model.predict(data_17['x_test']) 
y_prob = model.predict_proba(data_17['x_test'])[:,1]
results_18_17['results_model_inter'] = utf.compute_performance_stats(data_17['y_test'], y_pred, y_prob)
  

# TODO RUN FROM HERE
# compute SHAP values of model across train and test sets
explainer = shap.TreeExplainer(model, data_18['x_train'], feature_perturbation = "interventional", model_output='probability') # NB output='probability' decomposes inputs among Pr(Y=1='Attack'|X)
results_18_17['shap_18_train'] = explainer.shap_values(data_18['x_train'])
results_18_17['shap_18_val'] = explainer.shap_values(data_18['x_validation'])
results_18_17['shap_18_test'] = explainer.shap_values(data_18['x_test'])
results_18_17['shap_17_test'] = explainer.shap_values(data_17['x_test'])   # takes 83 mins to do 1,565,519 explanations

# SAVE DATA 
pickle.dump(results_18_17, open(save_loc + "results_18_17_shap.pkl", "wb")) 
pickle.dump(model, open(save_loc + "model_18.pkl", "wb"))

# scale data using Sklearn minmax scaler, NB here we fit the scaler to both the train and val normal data
scaler = MinMaxScaler(feature_range=(-1,1))          
scaler.fit( np.concatenate((results_18_17['shap_18_train'], results_18_17['shap_18_val'][ np.where( data_18['label_validation'] == "Benign" )[0] ]), axis=0) );                         
shap_train_scaled = scaler.transform(results_18_17['shap_18_train'])   # scale the training set data
shap_val_scaled = scaler.transform(results_18_17['shap_18_val'])           # scale the validation set data
shap_test_A_scaled = scaler.transform(results_18_17['shap_18_test'])         # scale the test set data
shap_test_B_scaled = scaler.transform(results_18_17['shap_17_test'])         # scale the test set data

# before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
#x_data, val_data = train_test_split(shap_train_scaled, test_size=0.2, random_state=10)

x_data = shap_train_scaled.copy()
val_data = shap_val_scaled[ np.where( data_18['label_validation'] == "Benign" )[0] ]

# perform grid search to find the best paramters to use for the autoencoder model
# specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
"""
parameters = {
                                    # encoder params to search across
                                    'dense_1_units':[1472],                         'dense_1_activation':['relu'],  # 32/4
                                    'dense_2_units':[198],                          'dense_2_activation':['relu'],  # 16/2
                                    'dense_3_units':[24],                           'dense_3_activation':['relu'],  # 
                                    # decoder params to search across
                                    'dense_4_units':[750],  'dense_4_activation':['relu'],  # 16/2
                                    'dense_5_units':[864],                          'dense_5_activation':['relu'],  # 32/4
                                    'dense_6_units':[x_data.shape[1]],              'dense_6_activation':['tanh']    
                                }

# check size of grid space to ensure not too large
z = [*parameters.values()]                       # get values of each sublist in the overall parameter list 
z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 

# perform the grid search and return parameters of the best model
_, results_18_17['best_params'] = utf.get_hyper_Autoencoder(parameters, x_data, val_data, method='exact', num_epochs=10, batch_size=1024, AE_type = 'joint', loss_monitor='val')   

"""

results_18_17['best_params'] = [[], [], [1048, 172, 18, 816, 1480, 67], ['relu','relu','relu','relu','relu','tanh']] # Model params for normal val, xgboost depth = 6


# Using the best parameters, build and train the final model
AE_model_18 = utf.Autoencoder(x_data.shape[1], results_18_17['best_params'][2], results_18_17['best_params'][3], AE_type='joint', learning_rate=1e-3) # create the AE model object

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=5, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
history_AE_SHAP = AE_model_18.full.fit(x_data, x_data, epochs=1000, batch_size=1024, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
# plot the training curve
plt.plot(history_AE_SHAP["loss"], label="Training Loss")
plt.plot(history_AE_SHAP["val_loss"], label="Validation Loss")
plt.title("Training curve for Autoencoder trained on CICIDS2018")
plt.xlabel("No. of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# SAVE DATA 
AE_model_18.full.save_weights(save_loc + 'AE_model_18_weights')
    
ground_truth_locs = np.where(data_18['y_test'] == 1)[0]
# use the explanations to identify the new attack samples on the CICIDS17 Dataset (Intra)
temp, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model_18, x_data, shap_test_A_scaled, ground_truth_locs, plt_title='Autoencoder trained on SHAP values from CICIDS2018', threshold=95, new_attack_stats=False)
# get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
y_prob = model.predict_proba(data_18['x_test'])[:, 1]
y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
y_pred = (y_prob>0.5)*1
# compute final results
results_18_17['results_overall_intra'] = utf.compute_performance_stats(data_18['y_test'], y_pred, y_prob)


ground_truth_locs = np.where(data_17['y_test'] == 1)[0]
# use the explanations to identify the new attack samples on the CICIDS17 Dataset (Intra)
temp, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model_18, x_data, shap_test_B_scaled, ground_truth_locs, plt_title='Autoencoder trained on CICIDS2018 Applied to CICIDS2017', threshold=95, new_attack_stats=False)
# get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
y_prob = model.predict_proba(data_17['x_test'])[:, 1]
y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
y_pred = (y_prob>0.5)*1
# compute final results
results_18_17['results_overall_inter'] = utf.compute_performance_stats(data_17['y_test'], y_pred, y_prob)

# SAVE DATA 
pickle.dump(results_18_17, open(save_loc + "results_18_17_final.pkl", "wb"))

results_percentile_18 = pd.DataFrame()

for i in range(85, 99):
    
    _, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model_18, x_data, shap_test_B_scaled, ground_truth_locs, plt_title='Autoencoder - CICIDS2018 -> CICIDS2017 - Perc('+str(i)+')', threshold=i, new_attack_stats=False)
    # get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
    y_prob = model.predict_proba(data_17['x_test'])[:, 1]
    y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
    y_pred = (y_prob>0.5)*1
    # compute final results
    temp = utf.compute_performance_stats(data_17['y_test'], y_pred, y_prob)
    temp['Threshold'] = i
    results_percentile_18 = results_percentile_18.append( temp, ignore_index=True)

# SAVE DATA 
pickle.dump(results_percentile_18, open(save_loc + "results_percentile_18.pkl", "wb"))













# EoF