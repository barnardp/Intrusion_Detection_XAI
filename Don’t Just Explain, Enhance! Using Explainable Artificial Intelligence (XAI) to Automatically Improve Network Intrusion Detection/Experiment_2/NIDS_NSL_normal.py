# -*- coding: utf-8 -*-
"""
    Code that constructs baseline AE trained on raw dataset, ie doesnt use the explanations to enhance its performance. Nb the architecture used here
    corresponds to that from the paper: Improving Performance of Autoencoder-based Network Anomaly Detection on NSL-KDD dataset

    FINDINGS:   When the AE is trained using the paramters from the paper, ie using SGD, 50 epochs, MAE loss, mini-batching, 95th percentile threshold, the results are no where near what the paper presents. Instead
                the resulting Accuracy is 45%, Recall is 12%, Precision is 57% and F1 is 20%.
                
                When the same paramters are used, but using adam instead of SGD, the results are much better, with Accuracy: 80.7%, Recall: 84.8%, Precision: 81.9%, F1: 83.3%
                
    
    
"""


import os
os.chdir('D:/')                  	# location where files are stored (main + utility)
save_loc = 'D:/'                    # location to save results

data_loc = 'D:/Datasets/NSL_KDD/'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xgboost
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Flatten, Subtract, Multiply, Reshape, AveragePooling1D, Average
import shap
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import time

#import utility_funcs as utf

np.random.seed(10)
tf.random.set_seed(10)


def read_KDD(data_loc, create_bin=None, create_multi=None):
    """ ********************************** Read in KDD NSL Dataset and preprocess ********************************** """
    # names of each column, NB 2nd last column are intrustion type, last column is no of models that correctly predicts the label
    feature_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
                     'num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
                     'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
                     'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
                     'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','intrusion_type','difficulty']
    
    X_train = pd.read_csv(data_loc+'KDDTrain+.txt', names = feature_names, header=None)     # read in training data
    Y_train = X_train['intrusion_type'].copy()                                              # extract label column
    X_train = X_train.drop(['intrusion_type','difficulty'],axis=1)                          # drop the label and 'difficulty' columns from data
    
    X_test = pd.read_csv(data_loc+'KDDTest+.txt', names = feature_names, header=None)       # read in test data
    Y_test = X_test['intrusion_type'].copy()                                                # extract label column
    X_test = X_test.drop(['intrusion_type','difficulty'],axis=1)                            # drop the label and 'difficulty' columns from data
    
    feature_names.remove('intrusion_type')
    feature_names.remove('difficulty')
    
    # Encode the categorical features using one-hot encoder Sckit learn library
    enc = OneHotEncoder()
    enc.fit(X_train[['protocol_type', 'service', 'flag']])
    # encode the categorical features across the train set and store results into a temporary dataframe
    temp = pd.DataFrame(enc.transform(X_train[['protocol_type', 'service', 'flag']]).toarray(), columns=enc.get_feature_names(['protocol_type', 'service', 'flag']))
    # remove the original categorical features from the main train set 
    X_train = X_train.drop(['protocol_type', 'service', 'flag'], axis=1)
    # concatenate the temp dataframe to the main train set
    X_train = pd.concat([X_train, temp], axis=1)
    # REPEAT FOR THE TEST SET: encode the categorical features across the test set and store results into a temporary dataframe
    temp = pd.DataFrame(enc.transform(X_test[['protocol_type', 'service', 'flag']]).toarray(), columns=enc.get_feature_names(['protocol_type', 'service', 'flag']) )
    # remove the original categorical features from the main train set 
    X_test = X_test.drop(['protocol_type', 'service', 'flag'], axis=1)
    # concatenate the temp dataframew to the main train set
    X_test = pd.concat([X_test, temp], axis=1)
    # REPEAT for the feature_names list
    feature_names.remove('protocol_type')
    feature_names.remove('service')
    feature_names.remove('flag')
    feature_names = feature_names + list(enc.get_feature_names(['protocol_type', 'service', 'flag']))
        
    # get locs of new attacks in Test+ set
    label_encoder_all_train = dict(zip(Y_train.unique(), range(len(Y_train.unique()))))   # encoder for all unique labels seen during training
    temp = Y_test.copy()
    temp = temp.map(label_encoder_all_train)
    temp.isnull().sum().sum()/len(temp)             # count no of NANs as % of entire test set, NB 16.6 of Test set consists of new attacks
    new_attack_locs = np.where(temp.isnull())[0]    # keep track of where new attacks occur, need to detect these as shifts later
    
    # create binary encoded versions of the labels     
    Y_train_bin = np.zeros_like(Y_train)    
    Y_train_bin[np.where(Y_train != 'normal')[0]] = 1    
    Y_test_bin = np.zeros_like(Y_test)    
    Y_test_bin[np.where(Y_test != 'normal')[0]] = 1
    
    # create copy of the train set, but only include the normal instances
    locs = np.where(Y_train == 'normal')[0]
    X_train_normal = X_train.iloc[locs]
    
    # save all variables in a disctionary
    data = {}
    data['feature_names'] = feature_names
    data['X_train'] = X_train
    data['X_train_normal'] = X_train_normal
    data['Y_train'] = Y_train
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    data['Y_train_bin'] = Y_train_bin
    data['Y_test_bin'] = Y_test_bin
    data['new_attack_locs'] = new_attack_locs

    return data

def outlier_disposal(data):
    # Implement the outlier scheme from paper (algorithm 2)
    # calculate the 95th percentile of each feature
    percentile = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        percentile[i] = np.percentile(data[:,i], 95)
    
    # create a mask array which repeats the precentiles for each sample in the dataset
    mask = np.tile(percentile, (len(data),1))
    # apply the mask to the dataset, and sum across each row
    temp = np.sum(data>mask, axis=1)
    locs = np.where(temp == 0)[0] # locations that arent outliers
    
    return data[locs]
    

def normalise_data(x_train, x_test):

    x_min = x_train.min()
    x_max = x_train.max()
    
    x_train_std = (x_train-x_min)/(x_max - x_min) # NB In paper, authors do one more step (eq 7), however, as they use min, max=(0,1), this gives same result
    
    x_test_std = (x_test-x_min)/(x_max - x_min)
    
    return x_train_std, x_test_std



class Autoencoder(Model):
    """ class to create an autonecoder
    num_units:    		    numpy array with elements representing num of dense units for each layer
    act_fn:       		    list with each element corresponding to the activation function of that layer   		
    num_encoder_layers: 	number of layers in the encoder, ie remaining layers are for the decoder  
    """      
    
    def __init__(self, x_shape, num_units, act_fn, num_encoder_layers=3, seed=0):
        super(Autoencoder, self).__init__()
        self.x_shape = x_shape
        self.num_units = num_units
        self.act_fn = act_fn
        self.num_encoder_layers = num_encoder_layers # NB the last layer of the encoder corresponds to the latent layer
                
        tf.random.set_seed(seed)
        np.random.seed(seed)
       
        # define the encoder
        encoder_input = Input(shape=(self.x_shape,))
        layer_i = tf.keras.layers.Dense(self.num_units[0], activation=self.act_fn[0])(encoder_input)    # add the first hidden layer
        for i in range(1, self.num_encoder_layers):
            layer_i = tf.keras.layers.Dense(self.num_units[i], activation=self.act_fn[i])(layer_i)      # add the next hidden layer  
        # finalise the encoder
        self.encoder = Model(encoder_input, layer_i, name='encoder')
        	
        # define the decoder  - NB the last layer of the decoder corresponds to the final output of the AE
        decoder_input = Input(shape=(self.num_units[i],))
        layer_i = tf.keras.layers.Dense(self.num_units[i+1], activation=self.act_fn[i+1])(decoder_input)    # add the first hidden layer
        for i in range(i+2, len(self.num_units)):
            layer_i = tf.keras.layers.Dense(self.num_units[i], activation=self.act_fn[i])(layer_i)          # add the next hidden layer 
        # finalise the decoder
        self.decoder = Model(decoder_input, layer_i, name='decoder')
        		
        # now combine together into a single AE
        self.final_output = self.decoder(self.encoder(encoder_input))
        self.full = Model(encoder_input, self.final_output, name='full_AE')
        
        # define the loss function
        self.full.compile(optimizer='adam', loss='mse')  # Unlike whats described in the paper, SGD seems to result in very poor results, while adam optimiser gives okay results
        self.compile(optimizer='adam', loss='mse')
        
    def call(self, x, **kwargs):
        return self.full(x)

    

def compute_performance_stats(y_true, y_pred, y_prob=None, title=None, rnd=4):  
    
    # here, normal behaviour is encoded as class 0, anomaly/attack is encoded as class 1
    conf_matrix = metrics.confusion_matrix(list(y_true), list(y_pred))
    
    tn, fp, fn, tp = conf_matrix.ravel() # where tp-> predicting a shift when indeed a shift occurs, fp-> predicting shift(y_pred=1), when no shift occurs
    
    TPR = tp/(tp+fn) # true positive rate / recall
    TNR = tn/(tn+fp) # true negative rate / specificity
    PREC =  tp/(tp+fp) # precision - how often a predicted anomaly really is an anomaly
    FNR = fn/(fn+tp) # false negative rate / miss rate
    FPR = fp/(fp+tn) # false positive rate
    TS = tp/(tp+fn+fp) # threat score / critical success rate
    ACC = (tp+tn)/(tp+fp+tn+fn) # accuracy
    F1 = 2 * (PREC * TPR) / (PREC + TPR)
    
        
    if y_prob is not None:
        # computing curves instead of just crisp numbers
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
        AUC = metrics.roc_auc_score(y_true, y_prob)
        AUPR = metrics.average_precision_score(y_true, y_prob)
        if title is not None:
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % AUC)
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve for " + title)
            plt.legend(loc="lower right")
            plt.show()
        return pd.DataFrame( np.round(np.array([[tn,fp,fn,tp,ACC,TPR,PREC,F1,FPR,TNR,FNR,TS, AUC, AUPR]]), rnd) , columns=['TN','FP','FN','TP','ACCURACY','RECALL','PRECISION','F1','FPR','TNR','FNR','T-SCORE','AUC','AUPR'])
    
    return pd.DataFrame( np.round(np.array([[tn,fp,fn,tp,ACC,TPR,PREC,F1,FPR,TNR,FNR,TS]]), rnd) , columns=['TN','FP','FN','TP','ACCURACY','RECALL','PRECISION','F1','FPR','TNR','FNR','T-SCORE'])


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = np.mean(grads, axis=0)
    return integrated_gradients

def interpolate_input(baseline, x_input, alphas):
    delta = x_input - baseline
    images = baseline + alphas[:, tf.newaxis] * delta
    return images

# Compute gradients using tf.GradientTape. NB as we can only explain one class (while the AE outputs multiple outputs for each feature), we
# first need to add another few layers to the tf model which will compute the average error and then classify the input as an anomaly or not
# NB can try apply elementwise threshold, ie the 95th percentile loss of each individual feature, then take average and explain that
def compute_gradients(model, x_input):
    with tf.GradientTape() as tape:
        tape.watch(x_input)
        x_recon = model(x_input)
        loss = tf.keras.losses.mean_squared_error(x_input, x_recon)
        #final_pred = tf.cast(tf.math.greater_equal(loss, threshold), dtype=tf.float32).numpy()             # NB might need to use a soft output approach, ie tf.subtract(loss, threshold)        
    return tape.gradient(loss, x_input)

def AE_anomaly_detection(autoencoder, train_data, test_data, ground_truth_locs, plt_title=None, threshold=95, new_attack_stats=True):    
    """ main function that applies autoencoder to the train and test sets and calculates performance of detector 
        NB this function isnt affected by whether an attack (or normal class) is defined as being 1 or 0, as it creates its own 
        arrays (where attack:1, normal:0) based purely on the locations where the attacks occur/are predicted
    """
    
    # To find anomalies, first compute Reconstruction Error on train data
    #encoded_train = autoencoder.encoder.predict(train_data)
    #decoded_train = autoencoder.decoder.predict(encoded_train)
    decoded_train = autoencoder(train_data)
    mse_train = np.mean(np.abs((train_data - decoded_train)), axis=1)   # np.mean(np.power(train_data - decoded_train, 2), axis=1)
    # Then calculate error threshold based on RE distribution 
    error_threshold = np.percentile(mse_train, threshold)    #np.median(mse_train)+3*np.std(mse_train)  # np.max(mse_train) 
    
    # Now compute the RE across the test points
    encoded_test = autoencoder.encoder.predict(test_data)
    decoded_test = autoencoder.decoder.predict(encoded_test)
    #decoded_test = autoencoder(test_data)                            # NB this line performs the combination of the two lines above, but may fail due to requiring higher memory since it invovles the entire DNN at once
    mse_test = np.mean(np.abs((test_data - decoded_test)), axis=1)
    # get locs of all test points above the threshold, i.e., anomalies
    pred_new_attack_locs = np.where(mse_test>error_threshold)[0]
    
    # compute performance metrics on autoencoder based anomaly detector - NB normal behaviour is class 0, atacks/new attacks is class 1 
    y_true = np.zeros(len(test_data),)  
    y_true[ground_truth_locs] = 1
    normal_locs = np.where(y_true == 0)[0]
    
    performance_summary = None
    if new_attack_stats == True: # calculate the performance of our anomaly detector to detect new attacks from (known attacks AND normal samples)
        y_pred = np.zeros(len(test_data),)
        y_pred[pred_new_attack_locs] = 1       
        performance_summary = compute_performance_stats(y_true, y_pred)
    
     
    # Plot MSE distributions 
    if plt_title != None:
        sns.kdeplot(data=mse_train, bw_adjust=1, log_scale=True, color='blue', label='Train set (Class: Normal)') # need to use log scale here as dist contains very high outliers
        y_max = plt.gca().get_ylim()[1]
        plt.vlines(error_threshold, ymin=0, ymax=y_max, colors='red', label='Threshold (' + str(threshold) + 'th Percentile)')
        sns.kdeplot(data=mse_test[normal_locs], bw_adjust=1, log_scale=True, color='orange', linestyle=':', linewidth=2, label='Test set (Class: Normal)')
        sns.kdeplot(data=mse_test[ground_truth_locs], bw_adjust=1, log_scale=True, color='green', label='Test set (Class: Attack)')
        plt.xlabel("MSE")
        plt.legend(loc='upper left')
        plt.title(plt_title)
        plt.show()
    
    
    return performance_summary, pred_new_attack_locs, error_threshold

def second_stage(attrib_train, attrib_test, xai_name, Y_test, y_pred, new_attack_locs, best_params, batch_size=512, min_delta=1e-5):
    """
        Autoencoder Params from our initial paper (XGBoost-SHAP, NSLKDD): [1456,724,14,632,1644,122]  
        Autoencoder Params found for (Autoencoder-DeepSHAP, NSLKDD): [2200, 300, 14, 480, 500, 122]
        Autoencoder Params found for (Autoencoder-Integrated Gradients, NSLKDD): [1280, 560, 14, 560, 1360, 122]
        
    """
    results = {}
    results[xai_name + '_train'] = attrib_train
    results[xai_name + '_test'] = attrib_test
    
    # scale data using Sklearn minmax scaler
    scaler = MinMaxScaler(feature_range=(-1,1))                                     
    results[xai_name + '_train_scaled'] = scaler.fit_transform(attrib_train)  # scale the training set data
    results[xai_name + '_test_scaled'] = scaler.transform(attrib_test)        # scale the test set data
    
    # before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
    x_data, val_data = train_test_split(results[xai_name + '_train_scaled'], test_size=0.2, random_state=10) #  np.arange(5,18,1).tolist()
    
    #best_params = [[1456,724,14,632,1644,122], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  
    
    # Using the best parameters, build and train the final model
    AE_model = Autoencoder(x_data.shape[1], best_params[0], best_params[1], AE_type = 'joint') # create the AE model object
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=20, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
    results['history'] = AE_model.full.fit(x_data, x_data, epochs=1000, batch_size=batch_size, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
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
    results['performance_new_attacks'], results['new_attack_pred_locs'], _ = AE_anomaly_detection(AE_model, results[xai_name + '_train_scaled'], results[xai_name + '_test_scaled'], new_attack_locs, plt_title=('Autoencoder loss for ' + xai_name + ' based explanations'))
    
    # calculate overall accuracy of the IDS system (XGBoost IDS and Anomaly detector) to detect attacks
    results['y_pred_all'] = y_pred.copy() # np.zeros_like(y_pred) 
    results['y_pred_all'][results['new_attack_pred_locs']] = 1 
    results_combined = compute_performance_stats(Y_test, results['y_pred_all'])
    
    results['y_pred_all'] = np.zeros_like(y_pred) 
    results['y_pred_all'][results['new_attack_pred_locs']] = 1 
    results_XAI_only = compute_performance_stats(Y_test, results['y_pred_all']) # results when predictions of initial model are all completely overwritten by the XAI-AE layer
    
    return results_combined, results_XAI_only


""" Create Autoencoder

    - Only use the normal data from the train set to train the AE
    - best_params = [122-32-5-32-122] with ReLu used at all layers
    - Pre-processing: Outlier disposal --> min-max normalisation
    - MAE reconstruction error

"""

data_kdd = read_KDD(data_loc)
# Perform outlier detection. NB returns 39,252 samples (same as mentioned in paper)
x_train = outlier_disposal(data_kdd['X_train_normal'].values) 
# Perform normalisation on train and test data
x_train, x_test = normalise_data(x_train, data_kdd['X_test'].values)

NUM_TRIALS = 100
seed = 0                                    # seed used by TF
results_kf_model = pd.DataFrame()           # dataframe to keep track of the first stage performance at each trial
results_kf_IG_combined = pd.DataFrame()     # dataframe to keep track of the overall results when Integrated Gradients is used as the XAI technique
results_kf_IG_only = pd.DataFrame()         # dataframe to keep track of results when only the new attacks discovered by the second stage are classified as attacks, ie to compare against overall results
results_kf_SHAP = pd.DataFrame()            # dataframe to keep track of the overall results when Deep SHAP is used as the XAI technique
results_kf_SHAP_only = pd.DataFrame()       # dataframe to keep track of results when only the new attacks discovered by the second stage are classified as attacks
ig_time = []                                # record average time to compute IG on the train set at each fold
shap_time = []                              # record average time to compute DeepSHAP on the train set at each fold

# trial_n = 1
for trial_n in range(16, 20):
    print(trial_n)
    # randomly split the train set into train and validation sets
    x_data, val_data = train_test_split(x_train, test_size=0.2, random_state=trial_n) 
    
    
    tf.random.set_seed(trial_n) # change Tensorflows random seed value
    # create Autoencoder model as described in paper
    AE_model = Sequential()
    AE_model.add(InputLayer(input_shape=(122, )))
    AE_model.add(Dense(122, activation='relu', name='input'))
    AE_model.add(Dense(32, activation='relu'))
    AE_model.add(Dense(5, activation='relu', name='latent_space'))
    AE_model.add(Dense(32, activation='relu'))
    AE_model.add(Dense(122, activation='relu', name='output'))
    # compile and fit the model
    AE_model.compile(optimizer='adam', loss='mse')
    history = AE_model.fit(x_data, x_data, epochs=50, batch_size=512, shuffle=True, validation_data=(val_data, val_data), verbose=2).history
    
    """
    # plot the training curve
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training curve for Trial #" + str(trial_n))
    plt.xlabel("No. of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    """
    # compute recontruction threshold on train set based on 95th percentile
    loss = np.mean(np.abs(x_train - AE_model(x_train)), axis=1)
    threshold = np.percentile(loss, 95)
    # now apply to test set
    loss = np.mean(np.abs(x_test - AE_model(x_test)), axis=1)
    locs = np.where(loss > threshold)[0]
    
    y_pred = np.zeros_like(data_kdd['Y_test_bin'])
    y_pred[locs] = 1
    
    results = compute_performance_stats(data_kdd['Y_test_bin'], y_pred)
    results_kf_model = results_kf_model.append(results, ignore_index=True)

    #pickle.dump(results_kf_model, open(save_loc + "results_N_trials_model.pkl", "wb")) 
      
    """ Use Integrated Gradients to Explain the Reconstruction Error of the model """
    
    # Establish a good baseline, ie one where the output prediction of the model is close to zero (for class 1, which is being explained in this case)
    loc = np.argmin( np.mean(np.abs(x_train - AE_model(x_train)), axis=1) )     # find the sample that gives the lowest reconstructino error
    baseline_x = x_train[loc].reshape(1,-1)                                     # use the sample with the lowest predicted score as our baseline
    
    # create a scaling vector that can later be used to interpolate between the baseline to each data sample
    m_steps = 1000
    alphas = np.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.
    
    #batch_size = 1000
    IG_train = np.zeros(x_train.shape)    
    IG_test = np.zeros(x_test.shape) 
    
    # compute Integrated Gradients on the train set - NB takes 4:21 mins
    dt = time.time()
    for i, input_x in enumerate(tqdm( tf.convert_to_tensor(x_train) )): # takes 0.0065 secs per explanation @ m=1000
        
        interpolated_x = interpolate_input(baseline_x, input_x, alphas)                     # compute paths along baseline to current input
        path_gradients = compute_gradients(AE_model, interpolated_x)   # compute the gradients along each path 
        IG_train[i] = integral_approximation(path_gradients).reshape(1,-1)                  # now compute reimann integral, and populate result into main array
        #if not (i % batch_size):                                                           # every 1k samples, save data
          #  pickle.dump([IG_train, i], open(save_loc + "IG_train.pkl", "wb"))
    dt = (time.time() - dt)/len(x_train)
    ig_time.append(dt)
    
    # compute Integrated Gradients on the test set - NB takes 
    for i, input_x in enumerate(tqdm( tf.convert_to_tensor(x_test) )): # takes 0.006513057 secs per explanation @ m=1000
        
        interpolated_x = interpolate_input(baseline_x, input_x, alphas)                     # compute paths along baseline to current input
        path_gradients = compute_gradients(AE_model, interpolated_x)   # compute the gradients along each path 
        IG_test[i] = integral_approximation(path_gradients).reshape(1,-1)                  # now compute reimann integral, and populate result into main array

    
    # Implement second stage of the pipeline & calculate overall results
    # best_params = [[1280, 560, 14, 560, 1360, 122], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]
    best_params = [[1200, 570, 12, 680, 700, 122], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]
    temp_1, temp_2 = second_stage(IG_train, IG_test, xai_name='IG', Y_test=data_kdd['Y_test_bin'], y_pred=y_pred, new_attack_locs=data_kdd['new_attack_locs'], best_params=best_params, batch_size=2048, min_delta=1e-8)
    
    results_kf_IG_combined = results_kf_IG_combined.append(temp_1, ignore_index=True)
    results_kf_IG_only = results_kf_IG_only.append(temp_2, ignore_index=True)
   
    #pickle.dump([results_kf_IG_combined, results_kf_IG_only], open(save_loc + "results_N_trials_IG.pkl", "wb"))    
    
    """ Use DeepSHAP to explaine model """

    # Use DeepSHAP to explain predictions...NB Here, we create copy of the autoencoder model, but add an extra layer at the end which combines all the reconstructed outputs
    # into a single output representing the total loss of the model...which we then explain/decompose into 122 SHAP values for each instance being explained
    # create a copy of the original Autoencoder model
    layer_0 = Input(shape=(122,), name='input')                             # layer 0
    layer_1 = Dense(122, activation='relu', name='layer_1')(layer_0)        # layer 1
    layer_2 = Dense(32, activation='relu', name='layer_2')(layer_1)         # layer 2
    layer_3 = Dense(5, activation='relu', name='layer_3')(layer_2)          # layer 3
    layer_4 = Dense(32, activation='relu', name='layer_4')(layer_3)         # layer 4
    layer_5 = Dense(122, activation='relu', name='layer_5')(layer_4)        # layer 5
    # Add extra layers that cobine to compute and the loss of the model...NB here we split the computation into 3 seperate operations/ANN layers
    layer_subtract = Subtract()([layer_0, layer_5])                             # layer 6
    layer_square = Multiply()([layer_subtract, layer_subtract])                 # layer 7
    layer_mean = Dense(1, activation='relu', name='layer_mean')(layer_square)   # layer 8
    # Finally, combine into a single model
    AE_loss_model = Model(layer_0, layer_mean)
    
    # Load the pre-trained weights of the original autoencoder into the first 5 layers of the loss model
    #AE_loss_model.layers[:-1].load_weights('{}/temp_AE_weights'.format(save_loc));
    AE_loss_model.layers[1].set_weights( AE_model.layers[0].get_weights() ) # 122
    AE_loss_model.layers[2].set_weights( AE_model.layers[1].get_weights() ) # 32
    AE_loss_model.layers[3].set_weights( AE_model.layers[2].get_weights() ) # 5
    AE_loss_model.layers[4].set_weights( AE_model.layers[3].get_weights() ) # 32
    AE_loss_model.layers[5].set_weights( AE_model.layers[4].get_weights() ) # 122
    # Manually define the weight and bias of the mean layer, ie so that it computes the average of the squared error
    weights_mean = [np.ones((122,1))/122, np.zeros(1)] 
    AE_loss_model.layers[8].set_weights( weights_mean ) 
    
    
    """
    # check that the partial weights are correctly transferred to the loss model
    z = AE_model.predict(x_data[:5])
    intermediate_loss_model = Model(inputs=AE_loss_model.input, outputs=AE_loss_model.get_layer('layer_5').output)
    zz = intermediate_loss_model.predict(x_data[:5])
    if np.array_equal(z, zz):
        print("Partial weights correctly transferred to loss model")
    
    # check that loss model correctly outputs the loss of the autoencoder
    
    def model_loss(model, x_input): 
        x_recon = model(x_input)
        loss = tf.keras.losses.mean_squared_error(x_input, x_recon)
        return loss.numpy().reshape(-1,1)
    
    z = model_loss(AE_model, x_data[:5])
    zz = AE_loss_model.predict(x_data[:5])
    if np.allclose(z, zz):    # NB As we use tensorflows in built method to compute the AE loss, we can expect small rounding differences
        print("loss model correctly outputs the loss of the autoencoder")
    
    """
    
    # select a set of background examples to take an expectation over
    np.random.seed(trial_n)
    background_SHAP = x_train[np.random.choice(x_data.shape[0], 1000, replace=False)]   
    explainer = shap.DeepExplainer(AE_loss_model, background_SHAP) # explainer = shap.DeepExplainer((AE_model.layers[0].input, AE_model.layers[-1].output), background_SHAP) ... Produces exact same results
    
    dt = time.time()
    shap_train = explainer.shap_values(x_data)[0] # takes 0.0027 for 10 samples, 0.002686 for 100 samples, 0.0046 for 1k samples, 0.03345 for 10k samples
    dt = (time.time() - dt)/len(x_train)
    shap_time.append(dt)
    
    shap_test = explainer.shap_values(x_test)[0] # NB for some reason original SHAP library fails on sample 5531 (error too large). To we overcome this we manually tweeked the tolerance in line 329 of file shap\explainers\_deep\deep_tf.py"
    """
    # check sum of shap values plus baseline adds up to the actual loss
    z = model_loss(AE_model, x_data)
    zz = (np.sum(shap_train, axis=1) + explainer.expected_value.numpy()).reshape(-1,1)
    if np.allclose(z, zz):    # NB As we use tensorflows in built method to compute the AE loss, we can expect small rounding differences
        print("SHAP values accurately decompose the loss on the train set")
    """
    # Implement second stage of the pipeline & calculate overall results
    best_params = [[2200, 300, 14, 480, 500, 122], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]
    temp_1, temp_2 = second_stage(shap_train, shap_test, xai_name='SHAP', Y_test=data_kdd['Y_test_bin'], y_pred=y_pred, new_attack_locs=data_kdd['new_attack_locs'], best_params=best_params)
    
    
    results_kf_SHAP = results_kf_SHAP.append(temp_1, ignore_index=True)
    results_kf_SHAP_only = results_kf_SHAP_only.append(temp_2, ignore_index=True)
   
    pickle.dump([results_kf_SHAP, results_kf_SHAP_only], open(save_loc + "results_N_trials_SHAP.pkl", "wb"))   
    

    
# results_kf_SHAP, results_kf_SHAP_only = pickle.load(open(save_loc + "results_N_trials_SHAP.pkl", "rb"))

#pickle.dump([results_kf_model, results_kf_IG_combined, results_kf_IG_only], open(save_loc + "results_N_trials.pkl", "wb"))     
# results_kf_model, results_kf_IG_combined, results_kf_IG_only = pickle.load(open(save_loc + "results_kf.pkl", "rb"))    
    


# now we can process the final results to reveal the average and standard deviation
#std_model = np.round(np.std(results_kf_model, axis=0),2)
#mean_model = np.round(np.mean(results_kf_model, axis=0),3)

# create box plots of the results ... TODO Also plot the results quoted from the paper
plt.figure()
results_kf_model.boxplot()
plt.title('Performance of original solution')
plt.grid(b=None)
plt.ylim(0, 1)

plt.figure()
results_kf_IG_combined.boxplot()
plt.title('Performance of XAI-Aided solution')
plt.grid(b=None)
plt.ylim(0, 1)

"""
plt.figure()
results_kf_IG_only.boxplot()
plt.title('Performance of XAI-Aided solution')
plt.grid(b=None)
plt.ylim(0, 1)
"""

plt.figure()
results_kf_SHAP.boxplot()
plt.title('Performance of SHAP-Aided solution')
plt.grid(b=None)
plt.ylim(0, 1)



# EoF