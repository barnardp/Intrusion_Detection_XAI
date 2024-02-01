# -*- coding: utf-8 -*-
"""

Overview of Code:   This code is an extension of the original work presented in "Robust Network Intrusion Detection through Explainable Artificial Intelligence (XAI)". In the
                    original work, the NIDS was contructed using SHAP explanatiosn. Here, we examine how well our system model generalises to other commonly used explanation
                    (XAI) methodlogies, such as Saabas, LIME MAPLE etc.
                    
                    
Steps performed in code:    - Read in NSL-KDD train and test sets, as well as previously trained XGBoost model.
                            - Also read in the original SHAP values, the associated autoencoder and performance results.
                            - For each new XAI method considered, do the following:
                                - compute explanations on the train and test set
                                - Train an autoencoder on the train set explanations
                                - Use the reconstructin error on the test set explanations to identify new attacks
                                - Compute performance metrics and save data
                            

"""



import os
os.chdir('D:/')              # location where files are stored (main + utility)
save_loc = 'D/:'    # location to save results
data_loc = 'D:/'                                                               # location of dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xgboost
import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from scipy.special import expit # Import logit function for base value transformation (convert SHAP values from log loss to probability space)
import time
import lime

from tqdm import tqdm

import utility_funcs as utf

    
def shapley_local_acc(y_model, shap_values,  shap_null, plot_dist=False):
    """
        function to compute average normalised standard deviation across predictions using Eqs 6 - 14 from SHAP trees paper:
        y_model:        output predictins of the model being explained
        shap_values:      correspnding shapley values of each feature
        shap_null:      the mean output of the model across the background set used prev to compute the shapley values
                        i.e. shap_null = model.predict(x_train[:background_size]).mean()
    """
   
    sigma = np.sqrt( np.mean(np.square( y_model - (np.sum(shap_values , axis=1) +  shap_null) ) ) ) / np.sqrt( np.mean( np.square(y_model) ) )
    
    if plot_dist is True:
        temp = np.sqrt( np.square( y_model - (np.sum(shap_values , axis=1) +  shap_null) ) ) / np.sqrt( np.mean( np.square(y_model) ) )
        plt.hist(temp, bins=20)
        plt.xlabel('normalised standard deviation between Shapley values and predictions')
    
    if sigma < 10E-6:
        ans = 1.0
    elif sigma < 0.01:
        ans = 0.9
    elif sigma < 0.05:
        ans = 0.75
    elif sigma < 0.1:
        ans = 0.6
    elif sigma < 0.2:
        ans = 0.4
    elif sigma < 0.3:
        ans = 0.3
    elif sigma < 0.5:
        ans = 0.2
    elif sigma < 0.7:
        ans = 0.1
    else:
        ans = 0.0
        
    return ans, sigma   

np.random.seed(10)
tf.random.set_seed(10)

""" *********************************** LOAD DATA ********************************** """

data_kdd = pickle.load(open(save_loc + "data_nsl.pkl", "rb"))
model = pickle.load(open(save_loc + "model.pkl", "rb"))
results_model = pickle.load(open(save_loc + "results_model.pkl", "rb"))

results_SHAP = pickle.load(open(save_loc + "results_SHAP.pkl", "rb"))
results_LIME_disc = pickle.load(open(save_loc + "results_LIME_disc.pkl", "rb"))
results_LIME_cont = pickle.load(open(save_loc + "results_LIME_cont.pkl", "rb")) 
results_sabaas = pickle.load(open(save_loc + "results_sabaas.pkl", "rb")) 
results_lime_IG_5 = pickle.load(open(save_loc + "results_lime_IG_5.pkl", "rb")) 
results_maple = pickle.load(open(save_loc + "results_maple.pkl", "rb"))

results_cfl = pickle.load(open(save_loc + "results_cfl.pkl", "rb"))
# compute area based metrics for tree shap solution from previous work
#y_pred = model.predict(data_kdd['X_test']) 
#y_prob = model.predict_proba(data_kdd['X_test'])[:,1]
#y_prob[results_SHAP['new_attack_pred_locs']] = 1.0 # update the new attacks
#y_pred = (y_prob>0.5)*1
#results_SHAP['performance_overall'] = utf.compute_performance_stats(data_kdd['Y_test_bin'], y_pred, y_prob)
#pickle.dump(results_maple, open(save_loc + "results_maple.pkl", "wb")) 


"""
autoencoder_shap = utf.Autoencoder(results_SHAP['x_data'].shape[1], results_SHAP['best_params'][2], results_SHAP['best_params'][3]) # create the AE model object
autoencoder_shap.full.load_weights('{}/AE_shap_weights'.format(save_loc)); 

decoded_train = autoencoder_shap(results_SHAP['shap_train_scaled'])
mse_train = np.mean(np.abs((results_SHAP['shap_train_scaled'] - decoded_train)), axis=1)   # np.mean(np.power(train_data - decoded_train, 2), axis=1)
# Then calculate error threshold based on RE distribution 
error_threshold = np.percentile(mse_train, 95)    #np.median(mse_train)+3*np.std(mse_train)  # np.max(mse_train) 

# Now compute the RE across the test points
encoded_test = autoencoder_shap.encoder.predict(results_SHAP['shap_test_scaled'])
decoded_test = autoencoder_shap.decoder.predict(encoded_test)
#decoded_test = autoencoder_shap(test_data)                            # NB this line performs the combination of the two lines above, but may fail due to requiring higher memory since it invovles the entire DNN at once
mse_test = np.mean(np.abs((results_SHAP['shap_test_scaled'] - decoded_test)), axis=1)
# get locs of all test points above the threshold, i.e., anomalies
pred_new_attack_locs = np.where(mse_test>error_threshold)[0]

# compute performance metrics on autoencoder based anomaly detector - NB normal behaviour is class 0, atacks/new attacks is class 1 
#y_true = np.zeros(len(test_data),)  
#y_true[ground_truth_locs] = 1
#normal_locs = np.where(y_true == 0)[0]
    
sns.kdeplot(data=mse_train, bw_adjust=1, log_scale=True, color='blue', shade=True, label='Train') # need to use log scale here as dist contains very high outliers
y_max = plt.gca().get_ylim()[1]
plt.vlines(error_threshold, ymin=0, ymax=y_max, colors='red', label='')
#sns.kdeplot(data=mse_test, bw_adjust=1, log_scale=True, color='orange', linestyle=':', linewidth=2, label='Test set (Class: Normal)')
sns.kdeplot(data=mse_test[data_kdd['new_attack_locs']], bw_adjust=1, log_scale=True, shade=True, color='green', label='Zero-Day')
plt.xlabel("Reconstruction Error")
#plt.legend(loc='upper right')
#plt.title('Reconstruction Error')
plt.show()
"""







""" ********************************** Compute Tree SHAP explanations ********************************** """
# NB Since we've previously already computed the SHAP values, the purpose here is simply to record the time taken to compute a single explanation (averaged across the train set) and the Local Accuracy of the explanations
explainer = shap.TreeExplainer(model, data_kdd['X_train'], feature_perturbation = "interventional", model_output='probability') # NB output='probability' decomposes inputs among Pr(Y=1='Attack'|X)
dt = time.time()
shap_train= explainer.shap_values(data_kdd['X_train'])  # takes 7 mins
dt = (time.time() - dt)/len(data_kdd['X_train']) # takes 0.00339 secs on average per explanation

test_idx = np.random.choice(len(data_kdd['X_train']), size=100, replace=False)  # choose 100 random samples from the train set - will use to compute local accuracy metric on the explanations
shapley_local_acc(model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1], results_SHAP['shap_train'][test_idx],  shap_null=explainer.expected_value) # compute the local accuracy

# compute y probably and then area based metrics
y_pred = model.predict(data_kdd['X_test']) 
y_prob = model.predict_proba(data_kdd['X_test'])[:,1]
y_prob[results_SHAP['all_attack_pred_locs']] = 1.0
results_SHAP['performance_overall']  = utf.compute_performance_stats(data_kdd['Y_test_bin'], results_SHAP['y_pred_all'], y_prob)

""" Calculate Saabas values: 
    
    NB Saabas values are Locally Accurate but Inconsistent (places too much weight on lower splits in the tree.)
    
    see: https://github.com/slundberg/shap/issues/323   https://arxiv.org/pdf/1706.06060.pdf    https://arxiv.org/pdf/2010.06734.pdf
    
"""

def shap_transform_scale(shap_values, expected_value, model_prediction):
    # Function to approximately transform the SHAP values in logit form into probability form
    # Adapted from jmmonteiro: https://github.com/slundberg/shap/issues/29        
    # NB model_prediction variable refers to the prediction probability returned by the model (e.g. if the XGB model is 99.96% certain that the instance belongs to class 3, then  model_prediction = 0.9996)       
    # NB To double check which class the original SHAP values decomposed, compute the average probability of each class (call to model.predict_proba) and compare against expected_value_transformed 
    # Once the SHAP values have been transformed with this function, they can plotted as follows: shap.force_plot(shap_values_transformed, test_set)
        
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    expected_value_transformed = expit(expected_value)   # equalivant to shap.links._logit_inverse( expected_value )
    
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = np.sum(shap_values, axis=1)
    
    #Computing the distance between the model_prediction and the transformed base_value
    #     distance_to_explain = abs(model_prediction - expected_value_transformed)
    distance_to_explain = model_prediction - expected_value_transformed
    
    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain
    
    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient[:,None]
    
    return shap_values_transformed, expected_value_transformed


explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")                                                           # NB saabas values are explained in terms of logloss and uses the data passed through each node during training as the background
dt = time.time()
saabas_train = explainer.shap_values(data_kdd['X_train'], approximate=True)    # NB using the approximate=True param here results in calculating the Saabas values on log loss form
saabas_train, saabas_expected_value = shap_transform_scale(saabas_train, explainer.expected_value, model.predict_proba(data_kdd['X_train'])[:, 1] ) # convert Saabas values from log loss form to normal probability scale
dt = (time.time() - dt)/len(data_kdd['X_train'])

saabas_test = explainer.shap_values(data_kdd['X_test'], approximate=True)
saabas_test, _ = shap_transform_scale(saabas_test, explainer.expected_value, model.predict_proba(data_kdd['X_test'])[:, 1] )

# pass the explanations to the second stage of the NIDS and calculate overall performance
results_sabaas, AE_saabas  = utf.second_stage(saabas_train, saabas_test, "saabas", data_kdd['X_test'], data_kdd['Y_test_bin'], results_model['Y_pred_prob_test'], data_kdd['new_attack_locs'])

# compute the local accuracy
shapley_local_acc(model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1], results_sabaas['saabas_train'][test_idx],  shap_null=saabas_expected_value)
results_sabaas['expected_value'] = saabas_expected_value         
# SAVE DATA
pickle.dump(results_sabaas, open(save_loc + "results_sabaas.pkl", "wb")) 
AE_saabas.full.save_weights(save_loc + 'AE_saabas_weights')    
   
""" ********************************** Compute LIME explanations (Discretized) ********************************** """
""" 
    NB we change the following values from their defaults:
        - num_features: maximum number of features present in explanation                                                           [CHANGED FROM 10 TO 41]
        - discretize_continuous: if True, all non-categorical features will be discretized into quartiles.                          [CHANGED FROM TRUE TO FALSE]
        - sample_around_instance: if True, will sample continuous features in perturbed samples from a normal centered at the
                                  instance being explained. Otherwise, the normal is centered on the mean of the feature data.      [CHANGED FROM False TO TRUE]       
        - feature_selection: feature selection method. can be 'forward_selection', 'lasso_path', 'none' or 'auto'                   [CHANGED FROM 'auto' TO 'none']                                        
"""

# create lime explainer object - dont use discretizer as we want explanations to be comparable to SHAP, use feature_Selectio='none' to avoid mixing lime weights from feature names and model coeffs
explainer_lime = lime.lime_tabular.LimeTabularExplainer(data_kdd['X_train'].values, training_labels=data_kdd['Y_train_bin'].values, feature_names=data_kdd['feature_names'], class_names=['Pr(Y=0=Normal|X)', 'Pr(Y=1=Attack|X)'], mode='classification', feature_selection='none',
                                                        discretize_continuous=True, discretizer='decile', random_state=100, verbose=False, sample_around_instance=True, kernel=None, kernel_width=None)

# seed = explainer_lime.random_state.get_state()
# explainer_lime.random_state.set_state(seed)
batch_size = 1000
start_idx = 0
LIME_train_decile = np.zeros(data_kdd['X_train'].shape) 
LIME_test_decile = np.zeros(data_kdd['X_test'].shape) 

dt = time.time()
for i in tqdm(range(start_idx, data_kdd['X_train'].shape[0])): # Explain the full train set - takes 35 hours 
    
    exp = explainer_lime.explain_instance(data_row=data_kdd['X_train'].iloc[i].values, predict_fn=model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None)  # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
    LIME_train_decile[i] = exp.reg_model.coef_ 
       
    if not (i % batch_size): # every 1k samples
        # SAVE DATA
        pickle.dump([LIME_train_decile, i], open(save_loc + "LIME_train_decile.pkl", "wb"))
dt = (time.time() - dt)/len(data_kdd['X_train'])

for i in tqdm(range(0, data_kdd['X_test'].shape[0])): # takes 
    exp = explainer_lime.explain_instance(data_row=data_kdd['X_test'].iloc[i].values, predict_fn=model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None)   # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
    #LIME_test_decile[i] = exp.reg_model.coef_
    LIME_test_decile[i] = exp['reg_model'].coef_.reshape(-1,)
    if not (i % batch_size): # every 1k samples
        # seed = explainer_lime.random_state.get_state()
        pickle.dump([LIME_test_decile, i], open(save_loc + "LIME_test_decile.pkl", "wb"))

del explainer_lime, batch_size, start_idx, exp, i

# pass the explanations to the second stage of the NIDS and calculate overall performance
results_LIME_disc, AE_lime_disc  = utf.second_stage(LIME_train_decile, LIME_test_decile, "lime", data_kdd['X_test'], data_kdd['Y_test_bin'], results_model['Y_pred_prob_test'], data_kdd['new_attack_locs'])
# compute y prob and then area based metrics
#y_pred = model.predict(data_kdd['X_test']) 
#y_prob = model.predict_proba(data_kdd['X_test'])[:,1]
#y_prob[results_LIME_disc['all_attack_pred_locs']] = 1.0
#results_LIME_disc['performance_overall']  = utf.compute_performance_stats(data_kdd['Y_test_bin'], results_LIME_disc['y_pred_all'], y_prob)
        
# compute the Local Accuracy of the explanations
temp_exps = np.zeros((100, 41)) 
temp_null = np.zeros(100,) 
for idx, i in enumerate(test_idx):  
    exp = explainer_lime.explain_instance(data_row=data_kdd['X_train'].iloc[i].values, predict_fn=model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None)  # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
    temp_exps[idx] = exp.reg_model.coef_ 
    temp_null[idx] = exp.reg_model.intercept_ 
    
shapley_local_acc(model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1], temp_exps,  shap_null=temp_null)

# SAVE DATA
pickle.dump(results_LIME_disc, open(save_loc + "results_LIME_disc.pkl", "wb")) 
AE_lime_disc.full.save_weights(save_loc + 'AE_lime_disc_weights')    
   

""" ********************************** Compute LIME explanations (Continuous) ********************************** """

results_LIME_cont = {} # compute LIME coefs where train set is NOT discretized

# create lime explainer object - dont use discretizer as we want explanations to be comparable to SHAP, use feature_Selectio='none' to avoid mixing lime weights from feature names and model coeffs
explainer_lime = lime.lime_tabular.LimeTabularExplainer(data_kdd['X_train'].values, training_labels=data_kdd['Y_train_bin'].values, feature_names=data_kdd['feature_names'], class_names=['Pr(Y=0=Normal|X)', 'Pr(Y=1=Attack|X)'], mode='classification', feature_selection='none',
                                                        discretize_continuous=None, random_state=100, verbose=False, sample_around_instance=True, kernel=None, kernel_width=None)


batch_size = 10000
start_idx = 0
LIME_train_cont = np.zeros(data_kdd['X_train'].shape) 
LIME_test_cont = np.zeros(data_kdd['X_test'].shape)

dt = time.time()
for i in tqdm(range(start_idx, data_kdd['X_train'].shape[0])): # Explain the full train set - takes 17 mins
   
    exp = explainer_lime.explain_instance(data_row=data_kdd['X_train'].iloc[i].values, predict_fn=model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None)  # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
    LIME_train_cont[i] = exp.reg_model.coef_ 
  
    if not (i % batch_size): # every 1k samples
        pickle.dump([LIME_train_cont, i], open(save_loc + "LIME_train_cont.pkl", "wb"))
dt = (time.time() - dt)/len(data_kdd['X_train'])

for i in tqdm(range(0, data_kdd['X_test'].shape[0])): # takes 
    exp = explainer_lime.explain_instance(data_row=data_kdd['X_test'].iloc[i].values, predict_fn=model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None)   # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
    LIME_test_cont[i] = exp.reg_model.coef_
    
    if not (i % batch_size): # every 1k samples
        # seed = explainer_lime.random_state.get_state()
        pickle.dump([LIME_test_cont, i], open(save_loc + "LIME_test_cont.pkl", "wb"))

del explainer_lime, batch_size, start_idx, exp, i

# pass the explanations to the second stage of the NIDS and calculate overall performance
results_LIME_cont, AE_lime_cont  = utf.second_stage(LIME_train_cont, LIME_test_cont, "lime", data_kdd['X_test'], data_kdd['Y_test_bin'], results_model['Y_pred_prob_test'], data_kdd['new_attack_locs'])
    
# compute the Local Accuracy of the explanations
temp_exps = np.zeros((100, 41)) 
temp_null = np.zeros(100,) 
for idx, i in enumerate(test_idx):  
    exp = explainer_lime.explain_instance(data_row=data_kdd['X_train'].iloc[i].values, predict_fn=model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None)  # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
    temp_exps[idx] = exp.reg_model.coef_ 
    temp_null[idx] = exp.reg_model.intercept_ 
    
shapley_local_acc(model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1], temp_exps,  shap_null=temp_null) # delta of 0.82 implies that the LIME values are just short of registering on the scale
    
# SAVE DATA
pickle.dump(results_LIME_cont, open(save_loc + "results_LIME_cont.pkl", "wb")) 
AE_lime_cont.full.save_weights(save_loc + 'AE_lime_cont_weights')    
   

""" compute integrated gradients using LIME as the gradient estimator 
    
    See: https://www.unofficialgoogledatascience.com/2017/03/attributing-deep-networks-prediction-to.html
    
    NB The attributions from integrated gradients sum to the difference between the prediction scores of the input and the baseline, i.e., 
        SUM(attrib)=model.pred_prob(x) - model.pred_prob(baseline)....Implying f(x) = SUM(attrib) + f(baseline)

"""

# Establish a good baseline, ie one where the output prediction of the model is close to zero (for class 1, which is being explained in this case)
loc = np.argmin(model.predict_proba(data_kdd['X_train'].values)[:,1])
baseline_x = data_kdd['X_train'].values[loc]                                       # use the sample with the lowest predicted score as our baseline
y_baseline = model.predict_proba( data_kdd['X_train'].iloc[loc:loc+1] )[:,1]
# interpolate between the baseline to each data sample
m_steps=5
alphas = np.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = np.mean(grads, axis=0)
    return integrated_gradients

explainer_lime = lime.lime_tabular.LimeTabularExplainer(data_kdd['X_train'].values, training_labels=data_kdd['Y_train_bin'].values, feature_names=data_kdd['feature_names'], class_names=['Pr(Y=0=Normal|X)', 'Pr(Y=1=Attack|X)'], mode='classification', feature_selection='none',
                                                        discretize_continuous=None, random_state=100, verbose=False, sample_around_instance=True, kernel=None, kernel_width=None)

batch_size = 1000
curr_sample = np.zeros((m_steps+1, 41))   
lime_IG_5_train = np.zeros(data_kdd['X_train'].shape)    
lime_IG_5_test = np.zeros(data_kdd['X_test'].shape) 

dt = time.time()
for i, input_x in enumerate(tqdm( data_kdd['X_train'].values )): # takes 100 mins
    delta = input_x - baseline_x
    for num, m in enumerate(alphas): # calculate its interpolated version
        data_row = baseline_x + (m * delta)
        exp = explainer_lime.explain_instance(data_row, model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None) # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
        curr_sample[num] = exp.reg_model.coef_
    # now compute reimann integral, and populate result into main array
    lime_IG_5_train[i] = integral_approximation(curr_sample)
    if not (i % batch_size): # every 1k samples
        pickle.dump([lime_IG_5_train, i], open(save_loc + "lime_IG_5_train.pkl", "wb"))
dt = (time.time() - dt)/len(data_kdd['X_train'])

for i, input_x in enumerate(tqdm( data_kdd['X_test'].values )): # for each training sample
    delta = input_x - baseline_x
    for num, m in enumerate(alphas): # calculate its interpolated version
        data_row = baseline_x + (m * delta)
        exp = explainer_lime.explain_instance(data_row, model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None) # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
        curr_sample[num] = exp.reg_model.coef_  
    # now compute reimann integral, and populate result into main array
    lime_IG_5_test[i] = integral_approximation(curr_sample)
    if not (i % batch_size): # every 1k samples
        pickle.dump([lime_IG_5_test, i], open(save_loc + "lime_IG_5_test.pkl", "wb"))

# pass the explanations to the second stage of the NIDS and calculate overall performance
results_lime_IG_5, AE_lime_IG_5  = utf.second_stage(lime_IG_5_train, lime_IG_5_test, "lime_IG", data_kdd['X_test'], data_kdd['Y_test_bin'], results_model['Y_pred_prob_test'], data_kdd['new_attack_locs'])
        
# compute the Local Accuracy of the explanations
temp_exps = np.zeros((100, 41)) 
temp_null = np.zeros(100,) 
for idx, i in enumerate(test_idx):  
        
    input_x = data_kdd['X_train'].values[i]
    delta = input_x - baseline_x
    for num, m in enumerate(alphas): # calculate its interpolated version
        data_row = baseline_x + (m * delta)
        exp = explainer_lime.explain_instance(data_row, model.predict_proba, labels=(1,), top_labels=None, num_features=41, num_samples=1000, distance_metric='euclidean', model_regressor=None) # set the top_labels parameter to 1, so that explanation decomposes inputs among Pr(Y=1='Attack'|X)
        curr_sample[num] = exp.reg_model.coef_
        temp_null[idx] = exp.reg_model.intercept_ 
    # now compute reimann integral, and populate result into main array
    temp_exps[idx] = integral_approximation(curr_sample)
      
shapley_local_acc(model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1], temp_exps,  shap_null=y_baseline) # delta of 0.89 
    
z = model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1]
zz = np.sum(temp_exps, axis=1) + y_baseline

# SAVE DATA
pickle.dump(results_lime_IG_5, open(save_loc + "results_lime_IG_5.pkl", "wb")) 
AE_lime_IG_5.full.save_weights(save_loc + 'AE_lime_IG_5_weights')    
   

""" Maple Explanations 

    see https://github.com/GDPlumb/MAPLE    D:\anaconda3\envs\python_37\Lib\site-packages\shap\explainers\other
    https://blog.ml.cmu.edu/2019/07/13/towards-interpretable-tree-ensembles/
    
    At a high level, MAPLE uses the tree ensemble to identify which training points are most relevant to a new prediction and uses those points to fit a linear model that is used both 
    to make a prediction and as a local explanation
    
    When LIME defines its local explanations, it optimizes for the fidelity-metric with Nx set as a probability distribution centered on x. So we say it uses an unsupervised neighborhood 
    (which makes it poor at detecting global patterns): Near a global pattern, an unsupervised neighborhood will sample points on either side of it. Consequently, if the explanation is linear, 
    it will smooth the global pattern (i.e., fail to detect it). Importantly, the only indication that something might be awry is that the explanation will have lower fidelity.                                                                                                                                                                                            makes i.

"""

predictor = lambda x: model.predict_proba(x)[:, 1]

dt = time.time()
explainer = shap.explainers.other.Maple(predictor, data_kdd['X_train']) # black-Box MAPLE method NB takes 22.70 hours to finish
#explainer = shap.explainers.other.TreeMaple(model.predict, data_kdd['X_train'])  # Fast Tree based MAPLE, NB dosnt support ensemble tree classifiers yet
dt = (time.time() - dt)/60


# NB setting multiply_by_input to True will result in multiplying the learned coeffients by the mean-centered input, making the values roughly comparable to SHAP values.
# In order to return the intercept value as well, we first modify the _maple.py source file to return intercept[i] = exp[0] within the def attributions method
dt2 = time.time()
maple_train, _ = explainer.attributions(data_kdd['X_train'], multiply_by_input=False)      # NB takes 4.49 hours to complete
dt2 = (time.time() - dt2)/60

dt3 = time.time()
maple_test, _ = explainer.attributions(data_kdd['X_test'], multiply_by_input=False)        # NB takes 0.77 hours to complete
dt3 = (time.time() - dt3)/60


pickle.dump(results_maple, open(save_loc + "results_maple.pkl", "wb")) 
pickle.dump([dt, dt2, dt3], open(save_loc + "time_MAPLE.pkl", "wb")) 

# pass the explanations to the second stage of the NIDS and calculate overall performance
results_maple, AE_maple  = utf.second_stage(maple_train, maple_test, "maple", data_kdd['X_test'], data_kdd['Y_test_bin'], results_model['Y_pred_prob_test'], data_kdd['new_attack_locs'])

# SAVE DATA
pickle.dump(results_maple, open(save_loc + "results_maple.pkl", "wb")) 
pickle.dump([dt, dt2, dt3], open(save_loc + "time_MAPLE.pkl", "wb")) 
AE_maple.full.save_weights(save_loc + 'AE_maple_weights')    
   
# compute the Local Accuracy of the explanations
maple_temp, intercept = explainer.attributions(data_kdd['X_train'].iloc[test_idx], multiply_by_input=False) 
shapley_local_acc(model.predict_proba(data_kdd['X_train'].iloc[test_idx])[:,1], maple_temp,  shap_null=intercept) # delta value of 1.9 implies explanations are not accurate at all

"""
# perform same steps as done in experiment 3 of the original paper
predictor = lambda x: model.predict_proba(x)[:, 1]    
y_pred = predictor(data_kdd['X_train'])
x_data, val_data, y_data, y_val = train_test_split(data_kdd['X_train'].values, y_pred, test_size=0.2, random_state=10)    
explainer = shap.explainers.other._maple.MAPLE(x_data, y_data, val_data, y_val) 
for i in range(0, len(x_train)):
    e_maple = exp_maple.explain(x)
    coefs_maple = e_maple["coefs"]
X_train,    # np array
data_mean,  # (41,) mean of each feature
out_dim,    # 1
flat_out    # True
"""   
  

""" Compute RL Based Counterfactual 

    Results show no significant improvement in the overall NIDS, ie the error distributions of the AE are all similar accross the train, test and new attacks.
    One potential reason could be due to the fact the CFL method always scales the explanations according to the train set statisitics, potentially meaning that any 
    erroneous elements (within the output vector of the AE) would get clipped back to the train set range, thereby removing the error signal/information.
    
    Can compare against pure adversarial approach which doesnt perform any clipping etc. NB this can also be compared against pure gradient XAI methods, essentially
    showing the difference between methods that are in terms of the tangent of decision boundary (gradient) and those which are perpendicular to it (adversarial)
    
"""

results_counter = {}
# construct and train an autoencoder that can accurately reconstruct the raw input features
results_counter['scaler'] = MinMaxScaler(feature_range=(-1,1)) 
results_counter['X_train_scaled'] = results_counter['scaler'].fit_transform(data_kdd['X_train'])  # scale the training set data
results_counter['X_test_scaled'] = results_counter['scaler'].transform(data_kdd['X_test'])        # scale the test set data
# split the train set up into train and validation sets
x_data, val_data = train_test_split(results_counter['X_train_scaled'], test_size=0.2, random_state=10)

# [1400, 200, 6, 200, 250, 41]
results_counter['parameters'] = {
                                    # encoder params to search across
                                    'dense_1_units':[1400],                                   'dense_1_activation':['relu'],  # 32,64,128,256
                                    'dense_2_units':[250],                                 'dense_2_activation':['relu'],  # 32,48,64
                                    'dense_3_units':[6],            'dense_3_activation':['relu'], # 8,16, 24
                                    # decoder params to search across
                                    'dense_4_units':[200],                                        'dense_4_activation':['relu'],
                                    'dense_5_units':[200],                                    'dense_5_activation':['relu'],  # 64,128,256
                                    'dense_6_units':[41],               'dense_6_activation':['tanh']    
                                }

# check size of grid space to ensure not too large
z = [*results_counter['parameters'].values()]                       # get values of each sublist in the overall parameter list 
z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 

# perform the grid search and return parameters of the best model
_, results_counter['best_params'] = utf.get_hyper_Autoencoder(results_counter['parameters'], x_data, val_data, method='exact', num_epochs=15, batch_size=1024, AE_type = 'joint')   


# define the parameters of the encoder--decoder
AE_params = [[1400, 250, 17, 200, 200, 41], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  
AE_model = utf.Autoencoder(41, AE_params[0], AE_params[1], AE_type='joint') # create the AE model object        
# build and compile the model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=40, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
results_counter['history'] = AE_model.full.fit(x_data, x_data, epochs=1000, batch_size=1024, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
# plot the training curve
plt.plot(results_counter['history']["loss"], label="Training Loss")
plt.plot(results_counter['history']["val_loss"], label="Validation Loss")
plt.title("Training curve for Autoencoder trained on raw input space")
plt.xlabel("No. of epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# save data
pickle.dump(results_counter, open(save_loc + "results_counter.pkl", "wb")) 
AE_model.full.save(save_loc + "autoencoder_normal/", save_format="tf")
#AE_model.full.save_weights(save_loc + 'AE_model_normal_weights')    

AE_params = [[1400, 250,6, 200, 200, 41], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  
AE_model = utf.Autoencoder(41, AE_params[0], AE_params[1]) 
AE_model.full.load_weights(save_loc + 'AE_model_normal_weights'); 
# y_pred = AE_model.full.predict(data_kdd['X_test'])

# Load the counterfactual explanations
exp_cfl = pickle.load(open(save_loc + "exp_cfl.pkl", "rb"))


# pass the explanations to the second stage of the NIDS and calculate overall performance
results_cfl, AE_cfl  = utf.second_stage(exp_cfl['exp_train'], exp_cfl['exp_test'], "cfl", data_kdd['X_test'], data_kdd['Y_test_bin'], results_model['Y_pred_prob_test'], data_kdd['new_attack_locs'])
        
# SAVE DATA
pickle.dump(results_cfl, open(save_loc + "results_cfl.pkl", "wb")) 
AE_cfl.full.save_weights(save_loc + 'AE_cfl_weights')    
   
  
#EoF
