# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 06:07:59 2023

@author: barna
"""

""" PRE-TRAIN for DeepSHAP """

attrib_train = IG_train
attrib_test = IG_test
xai_name='IG'
Y_test=data_kdd['Y_test_bin']
new_attack_locs=data_kdd['new_attack_locs']

def second_stage(attrib_train, attrib_test, xai_name, Y_test, y_pred, new_attack_locs):
   
    results = {}
    results[xai_name + '_train'] = attrib_train
    results[xai_name + '_test'] = attrib_test
    
    # scale data using Sklearn minmax scaler
    scaler = MinMaxScaler(feature_range=(-1,1))                                     
    results[xai_name + '_train_scaled'] = scaler.fit_transform(attrib_train)  # scale the training set data
    results[xai_name + '_test_scaled'] = scaler.transform(attrib_test)        # scale the test set data
    
    # get all samples that are attacks on the test set BUT incorrectly classified by our AE model
    temp_test = np.where(data_kdd['Y_test_bin'] == 0)[0]
    temp_test = results[xai_name + '_test_scaled'][temp_test]
    
    # before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
    x_data2, val_data = train_test_split(results[xai_name + '_train_scaled'], test_size=0.2, random_state=10) #  np.arange(5,18,1).tolist()
    
    # perform grid search to find the best paramters to use for the autoencoder model
    # specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
    
    parameters = { # [1200, 570, 12, 680, 700, 122]
                                    # encoder params to search across
                                    'dense_1_units':[1200],                           'dense_1_activation':['relu'],  # 32,64,128,256
                                    'dense_2_units':[570],                            'dense_2_activation':['relu'],  # 32,48,64
                                    'dense_3_units':[12],                             'dense_3_activation':['relu'], # 8,16, 24
                                    # decoder params to search across
                                    'dense_4_units':[680],                            'dense_4_activation':['relu'],
                                    'dense_5_units':[700],                            'dense_5_activation':['relu'],  # 64,128,256
                                    'dense_6_units':[x_data2.shape[1]],               'dense_6_activation':['tanh']    
                                }
    
    # check size of grid space to ensure not too large
    z = [*parameters.values()]                       # get values of each sublist in the overall parameter list 
    z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 
    
    # perform the grid search and return parameters of the best model
    zz, best_params = utf.get_hyper_Autoencoder(parameters, x_data2, temp_test, method='exact', num_epochs=10, batch_size=2048, AE_type = 'joint', loss_monitor='val')   
    
    loc_best =  [item[1] for item in zz]
    plt.scatter(np.arange(len(loc_best)), loc_best)
    
    best_params = best_params[2:]
    #best_params = [[1456,724,14,632,1644,122], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  # use same params from our own paper
    
    # Using the best parameters, build and train the final model
    AE_model_IG = utf.Autoencoder(x_data2.shape[1], best_params[0], best_params[1], AE_type = 'joint') # create the AE model object
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
    results['history'] = AE_model_IG.full.fit(x_data2, x_data2, epochs=1000, batch_size=512, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
    # plot the training curve
    plt.plot(results['history']["loss"], label="Training Loss")
    plt.plot(results['history']["val_loss"], label="Validation Loss")
    plt.title("Training curve for " + xai_name)
    plt.xlabel("No. of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # perform anomaly detection based on the reconstruction error of the AE and save results
    # Nb here we try 2 different ways to mix the results from the initial predictions with those from the seconds stage (as unlike, xgboost thats good on seen data, the AE in the first stage isnt)
    results['performance_new_attacks'], results['new_attack_pred_locs'], _ = utf.AE_anomaly_detection(AE_model_IG, results[xai_name + '_train_scaled'], results[xai_name + '_test_scaled'], new_attack_locs, plt_title=('Autoencoder loss for ' + xai_name + ' based explanations'))
    
    # calculate overall accuracy of the IDS system (XGBoost IDS and Anomaly detector) to detect attacks
    results['y_pred_all'] = y_pred.copy() # np.zeros_like(y_pred) 
    results['y_pred_all'][results['new_attack_pred_locs']] = 1 
    results_kf_IG_combined = compute_performance_stats(Y_test, results['y_pred_all'])
    
    results['y_pred_all'] = np.zeros_like(y_pred)  
    results['y_pred_all'][results['new_attack_pred_locs']] = 1 
    results_kf_IG_only = compute_performance_stats(Y_test, results['y_pred_all'])
    
    return results_kf_IG_combined, results_kf_IG_only


""" PRE-TRAIN for DeepSHAP """
# temp_1, temp_2 = second_stage(shap_train, shap_test, xai_name='SHAP', Y_test=data_kdd['Y_test_bin'], y_pred=y_pred, new_attack_locs=data_kdd['new_attack_locs'])

attrib_train = shap_train
attrib_test = shap_test
xai_name='SHAP'
Y_test=data_kdd['Y_test_bin']
new_attack_locs=data_kdd['new_attack_locs']

def second_stage(attrib_train, attrib_test, xai_name, Y_test, y_pred, new_attack_locs):
   
    results = {}
    results[xai_name + '_train'] = attrib_train
    results[xai_name + '_test'] = attrib_test
    
    # scale data using Sklearn minmax scaler
    scaler = MinMaxScaler(feature_range=(-1,1))                                     
    results[xai_name + '_train_scaled'] = scaler.fit_transform(attrib_train)  # scale the training set data
    results[xai_name + '_test_scaled'] = scaler.transform(attrib_test)        # scale the test set data
    
    # before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
    x_data2, val_data = train_test_split(results[xai_name + '_train_scaled'], test_size=0.2, random_state=10) #  np.arange(5,18,1).tolist()
    
    # perform grid search to find the best paramters to use for the autoencoder model
    # specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
    
    parameters = { # [2200, 300, 14, 520, 680, 122]
                                    # encoder params to search across
                                    'dense_1_units':[2200],                                   'dense_1_activation':['relu'],  # 32,64,128,256
                                    'dense_2_units':[300],                                 'dense_2_activation':['relu'],  # 32,48,64
                                    'dense_3_units':[14],                                           'dense_3_activation':['relu'], # 8,16, 24
                                    # decoder params to search across
                                    'dense_4_units':[480],                                        'dense_4_activation':['relu'],
                                    'dense_5_units':[500],                                    'dense_5_activation':['relu'],  # 64,128,256
                                    'dense_6_units':[x_data2.shape[1]],               'dense_6_activation':['tanh']    
                                }
    
    # check size of grid space to ensure not too large
    z = [*parameters.values()]                       # get values of each sublist in the overall parameter list 
    z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 
    
    # perform the grid search and return parameters of the best model
    _, best_params = utf.get_hyper_Autoencoder(parameters, x_data2, val_data, method='exact', num_epochs=10, batch_size=512, AE_type = 'joint', loss_monitor='val')   
    
    
    best_params = best_params[2:]
    #best_params = [[1456,724,14,632,1644,122], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  # use same params from our own paper
    
    # Using the best parameters, build and train the final model
    AE_model_IG = utf.Autoencoder(x_data2.shape[1], best_params[0], best_params[1], AE_type = 'joint') # create the AE model object
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
    results['history'] = AE_model_IG.full.fit(x_data2, x_data2, epochs=1000, batch_size=512, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
    # plot the training curve
    plt.plot(results['history']["loss"], label="Training Loss")
    plt.plot(results['history']["val_loss"], label="Validation Loss")
    plt.title("Training curve for " + xai_name)
    plt.xlabel("No. of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # perform anomaly detection based on the reconstruction error of the AE and save results
    # Nb here we try 2 different ways to mix the results from the initial predictions with those from the seconds stage (as unlike, xgboost thats good on seen data, the AE in the first stage isnt)
    results['performance_new_attacks'], results['new_attack_pred_locs'], _ = utf.AE_anomaly_detection(AE_model_IG, results[xai_name + '_train_scaled'], results[xai_name + '_test_scaled'], new_attack_locs, plt_title=('Autoencoder loss for ' + xai_name + ' based explanations'))
    
    # calculate overall accuracy of the IDS system (XGBoost IDS and Anomaly detector) to detect attacks
    results['y_pred_all'] = y_pred.copy() # np.zeros_like(y_pred) 
    results['y_pred_all'][results['new_attack_pred_locs']] = 1 
    results_kf_IG_combined = compute_performance_stats(Y_test, results['y_pred_all'])
    
    results['y_pred_all'] = np.zeros_like(y_pred)  
    results['y_pred_all'][results['new_attack_pred_locs']] = 1 
    results_kf_IG_only = compute_performance_stats(Y_test, results['y_pred_all'])
    
    return results_kf_IG_combined, results_kf_IG_only































#