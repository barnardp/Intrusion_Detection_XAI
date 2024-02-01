# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:13:20 2021

Utility functions for KDD Intrusion (anomaly detection & XAI)
"""

import numpy as np
import pandas as pd

import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras import layers, Model, Input

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import random
import itertools
import warnings
import copy
from sklearn import metrics


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
    
    X_test_21 = pd.read_csv(data_loc+'KDDTest-21.txt', names = feature_names, header=None)  # read in test-21 data
    Y_test_21 = X_test_21['intrusion_type'].copy()                                          # extract label column
    X_test_21 = X_test_21.drop(['intrusion_type','difficulty'],axis=1)                      # drop the label and 'difficulty' columns from data
    
    feature_names.remove('intrusion_type')
    feature_names.remove('difficulty')
    
    # Encode each fo the categorical features using 'label encoding' technique 
    # Categorical features in the dataset include 'protocol_type', 'service', and 'flag'.
    # first, create encoders - dictionaries that map each feature value to a number, and keep mappings for test sets
    protocol_encoder = dict(zip(X_train['protocol_type'].unique(), range(len(X_train['protocol_type'].unique()))))
    service_encoder = dict(zip(X_train['service'].unique(), range(len(X_train['service'].unique()))))
    flag_encoder = dict(zip(X_train['flag'].unique(), range(len(X_train['flag'].unique()))))
    # now map each of the categorical columns to numbers
    X_train['protocol_type'] = X_train['protocol_type'].map(protocol_encoder)
    X_train['service'] = X_train['service'].map(service_encoder)
    X_train['flag'] = X_train['flag'].map(flag_encoder)
    
    # now repeat for the test data, but check that 'unseen' values dont result in NANs appearing in data
    X_test['protocol_type'] = X_test['protocol_type'].map(protocol_encoder)
    X_test['service'] = X_test['service'].map(service_encoder)
    X_test['flag'] = X_test['flag'].map(flag_encoder)
    
    # now repeat for the test-21 data, but check that 'unseen' values dont result in NANs appearing in data
    X_test_21['protocol_type'] = X_test_21['protocol_type'].map(protocol_encoder)
    X_test_21['service'] = X_test_21['service'].map(service_encoder)
    X_test_21['flag'] = X_test_21['flag'].map(flag_encoder)
    
    #X_test.isnull().values.any()               # No NANs found, so no need to flag early instances as 'shifted'
    
    # ALTERNATIVELY, can use label encoding method from Sckit learn
    enc = OrdinalEncoder()
    enc.fit(X_train[['protocol_type', 'service', 'flag']])
    #X_train[['protocol_type', 'service', 'flag']] = enc.transform(X_train[['protocol_type', 'service', 'flag']])
    #X_test[['protocol_type', 'service', 'flag']] = enc.transform(X_test[['protocol_type', 'service', 'flag']])
    #X_test_21[['protocol_type', 'service', 'flag']] = enc.transform(X_test_21[['protocol_type', 'service', 'flag']])
        
        
    # get locs of new attacks in Test+ set
    label_encoder_all_train = dict(zip(Y_train.unique(), range(len(Y_train.unique()))))   # encoder for all unique labels seen during training
    temp = Y_test.copy()
    temp = temp.map(label_encoder_all_train)
    temp.isnull().sum().sum()/len(temp)         # count no of NANs as % of entire test set
    new_attack_locs = np.where(temp.isnull())[0]   # keep track of where new attacks occur, need to detect these as shifts later
    
    # get locations of all new attacks in the Test-21 set
    temp = Y_test_21.copy()
    temp = temp.map(label_encoder_all_train)
    temp.isnull().sum().sum()/len(temp)         # count no of NANs as % of entire test set
    new_attack_locs_21 = np.where(temp.isnull())[0]
       
    # create binary encoded versions of the labels  
    Y_train_bin, Y_test_bin, Y_test_bin_21 = create_bin_KDD(Y_train, Y_test, Y_test_21)

    # create multi label encoded versions of the Y labels according to the 5 main classes - Normal, DoS, Probe, R2L, U2R
    Y_train_multi, Y_test_multi, Y_test_multi_21 = create_multi_KDD(Y_train, Y_test, Y_test_21)
       
    # get locations of all new attacks in the Train set , and locs of normal samples 
    train_attack_locs = np.where(Y_train_bin==1)[0]
    train_normal_locs = np.where(Y_train_bin==0)[0]
    
    # for each type of attack, isolate its samples in the test set
    new_attacks_by_type = {}
    new_attacks_by_type['DoS'] = np.intersect1d(np.where(Y_test_multi=='DoS')[0], new_attack_locs)
    new_attacks_by_type['Probe'] = np.intersect1d(np.where(Y_test_multi=='Probe')[0], new_attack_locs)
    new_attacks_by_type['R2L'] = np.intersect1d(np.where(Y_test_multi=='R2L')[0], new_attack_locs)
    new_attacks_by_type['U2R'] = np.intersect1d(np.where(Y_test_multi=='U2R')[0], new_attack_locs)

    
    """
    # snippet of code to analyse various characteristics of the NSL-KDD dataset, for example
    - check the unique labels in each train and test set, as original NSL Paper appears to be incorrect
       
    # from original paper, supposed to contain 21 attacks in train set plus 14 new in test set, but here we see there are 39 unique attacks in total
    z = np.unique(np.concatenate((Y_train.unique(), Y_test.unique()))) 
     
    
    in_train_not_test = np.array([]) # attacks that are in the train set BUT not in the test set
    in_test_not_train = np.array([]) # new attacks in test set
        
    for attack in z: #  here we see that the train set contains 2 attacks not seen in the test set, while the test set contains 17 new attacks
        if attack in Y_train.unique() and attack not in Y_test.unique():
            in_train_not_test = np.append(in_train_not_test, attack)
        if attack in Y_test.unique() and attack not in Y_train.unique():
            in_test_not_train = np.append(in_test_not_train, attack)
            
            
    # Check number of samples in KDDTest+ which are new attacks
    temp = Y_train.unique()
    new_attack_samples = [attack for attack in Y_test if attack not in temp] # 3750 samples are new attacks
    len(new_attack_samples)/len(Y_test)*100  # 16.6%
    
    # Check number of samples in KDDTest-21 which are new attacks
    new_attack_samples = [attack for attack in Y_test_21 if attack not in temp] # 3740 samples are new attacks
    len(new_attack_samples)/len(Y_test_21)*100  # 31.56%
            
    """
    
    # save all variables in a disctionary
    data = {}
    data['feature_names'] = feature_names
    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    data['X_test_21'] = X_test_21
    data['Y_test_21'] = Y_test_21
    data['Y_train_bin'] = Y_train_bin
    data['Y_test_bin'] = Y_test_bin
    data['Y_test_bin_21'] = Y_test_bin_21
    data['Y_train_multi'] = Y_train_multi
    data['Y_test_multi'] = Y_test_multi
    data['Y_test_multi_21'] = Y_test_multi_21
    data['new_attack_locs'] = new_attack_locs
    data['new_attack_locs_21'] = new_attack_locs_21
    data['train_attack_locs'] = train_attack_locs
    data['train_normal_locs'] = train_normal_locs
    data['new_attacks_by_type'] = new_attacks_by_type
    
    
    return data


def create_bin_KDD(Y_train=None, Y_test=None, Y_test_21=None):
    """  module to create a binary version of the label columns of the NSL KDD dataset  """
    
    temp = np.where(Y_train!='normal') # around 63% of the train set contains normal traffic
    Y_train_bin = Y_train.copy()
    Y_train_bin.iloc[temp]='attack'
    
    temp = np.where(Y_test!='normal')
    Y_test_bin = Y_test.copy()
    Y_test_bin.iloc[temp]='attack'
    
    temp = np.where(Y_test_21!='normal')
    Y_test_bin_21 = Y_test_21.copy()
    Y_test_bin_21.iloc[temp]='attack'
        
    label_encoder_bin = {'normal':0, 'attack':1}   # encoder for the binary label column
    
    Y_train_bin = Y_train_bin.map(label_encoder_bin)
    Y_test_bin = Y_test_bin.map(label_encoder_bin)
    Y_test_bin_21 = Y_test_bin_21.map(label_encoder_bin)
    
    return Y_train_bin, Y_test_bin, Y_test_bin_21
    

def create_multi_KDD(Y_train=None, Y_test=None, Y_test_21=None):
    """  create a multi-class version of the NSL KDD dataset  based on the 5 main type of attacks
    NB Training set should contain following number of instances per class:
        Normal (67,343), DoS (45,927), R2L (995), U2R (52), Probe (11,656)
    """
    main_multi_class_encoder = {} # classes based on those presented in: NETWORK INTRUSION DETECTION BASED ON ROUGH SET AND K-NEAREST NEIGHBOUR, Adebayo O. Adetunmbi
    for key in ['normal']:
        main_multi_class_encoder[key] = 'normal'
    for key in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'mailbomb']: # New attacks: apache2, udpstorm, processable, mailbomb
        main_multi_class_encoder[key] = 'DoS'       # DoS attacks
    for key in ['ipsweep', 'satan', 'nmap', 'portsweep', 'saint', 'mscan']: # New attacks: saint, mscan
        main_multi_class_encoder[key] = 'Probe' # Probe attacks
    for key in ['ftp_write', 'guess_passwd', 'warezmaster', 'warezclient', 'imap', 'phf', 'spy', 'multihop', 'named', 'xlock', 'sendmail', 'xsnoop', 'worm', 'snmpgetattack', 'snmpguess']: # New attacks: named, xclock, sendmail, xsnoop, worm, snmpgetattack, snmpguess
        main_multi_class_encoder[key] = 'R2L'  # R2L attacks
    for key in ['rootkit', 'loadmodule', 'buffer_overflow', 'perl', 'xterm', 'ps', 'sqlattack', 'httptunnel']: # New attacks: xterm, ps, sqlattack, httptunnel
        main_multi_class_encoder[key] = 'U2R'  # U2L attacks 
    
    Y_train_multi = Y_train.map(main_multi_class_encoder)
    Y_test_multi = Y_test.map(main_multi_class_encoder)
    Y_test_multi_21 = Y_test_21.map(main_multi_class_encoder)
    
    return Y_train_multi, Y_test_multi, Y_test_multi_21



class Autoencoder(Model):
    """ class to create an autonecoder
        num_units:    numpy array with elements representing num of dense units for each layer
        act_fn:       list with each element corresponding to the activation function of that layer
    """      
    def __init__(self, x_shape, num_units, act_fn, AE_type='joint', learning_rate=1e-3):
        super(Autoencoder, self).__init__()
        self.x_shape = x_shape
        self.num_units = num_units
        self.act_fn = act_fn
        #self.mae = tf.keras.losses.MeanAbsoluteError()
        
        tf.random.set_seed(10)
        np.random.seed(10)
        
        # define the encoder
        inputs = Input(shape=(self.x_shape,))
        encoder_front_end = tf.keras.Sequential([
            layers.Dense(self.num_units[0], activation=self.act_fn[0]),                                     # dense_1
            layers.Dropout(0.2),
            layers.Dense(self.num_units[1], activation=self.act_fn[1]),                                     # dense_2
            layers.Dense(self.num_units[2], activation=self.act_fn[2])                                      # dense_3
            ])(inputs)
        # finalise the encoder
        self.encoder = Model(inputs, encoder_front_end, name='encoder')
        
        # define the decoder 
        decoder_inputs = Input(shape=(self.num_units[2],))
        decoder_front_end = tf.keras.Sequential([
            layers.Dense(self.num_units[3], activation=self.act_fn[3]),                                     # dense_4
            layers.Dropout(0.2),
            layers.Dense(self.num_units[4], activation=self.act_fn[4]),                                     # dense_5
            layers.Dense(self.x_shape, activation=self.act_fn[5])                                           # dense_6
            ])(decoder_inputs)
        # finalise the decoder
        self.decoder = Model(decoder_inputs, decoder_front_end, name='decoder')
        
        # now combine together into a single AE
        self.final_output = self.decoder(self.encoder(inputs))
        self.full = Model(inputs, self.final_output, name='full_AE')
        
        if AE_type == 'random':
            self.encoder.trainable = False # if defining a random autoencoder, dont optimise the encoder weights
        
        # define the loss function
        #reconstruction_loss = self.mae(inputs, self.final_output)
        #self.full.add_loss(reconstruction_loss)
        self.full.compile(optimizer='adam', loss='mae')



    


def get_hyper_Autoencoder(parameters, x_data, val_data=None, method='exact', num_perm=None, num_epochs=20, batch_size=512, AE_type = 'joint'):
    """
        Method to perform grid search on an autoencoder:
        parameters:         Dictionary with search vectors of each layer (i) in the network, accepts 'dense_i_units', and 'dense_i_activation' similar to Sklearn 
        x_data:             Training dataset (numpy)
        val_data:           Validation dataset, if None, method forms this set from the training data
        method:             If 'exact', the entire grid space is searched UNLESS the sapce exceeds 'num_perm'. If 'random' samples 'num_perm' searches from space
        num_perm:           Can be used to set an upper bound of searches to perform, If total grid space exceeds this value, random sampling is 
                            performed automatically. OPTIONALLY, can be set to a fractional value (0,1) to search a fraction of the total grid space
        num_epochs:         The amount of epochs to train each model during the search space
        batch_size:         Batch size to use during each fit
        AE_type:            Specifies how the encoder and decoder are to be trained, use 'joint' if jointly trained, or 'random' for randomised/untrained encoder
    """    
    
    random.seed(1)
    tf.random.set_seed(1)
    
    grid_act_fn = []
    grid_num_units = []
    check_list = []
    results = [] # store results of each model as a sublist, where sublist[0]: training loss, sublist[1]: validation loss, sublist[3]: dense units params, sublist[4]: activation params
    
    if val_data is None:
        x_data, val_data = train_test_split(x_data, test_size=0.2, random_state=1)
        
    z = [*parameters.values()]                                          # get values of each sublist in the overall parameter list 
    total_perm = np.prod(np.array([len(sublist) for sublist in z]))     # total number of permutations in the grid 
    
    if isinstance(num_perm, float): # OPTIONALLY, if num permutations is given as % of the total grid space
        num_perm = int(num_perm*total_perm) # convert to actual number of searches to perform
    
    if isinstance(num_perm, int) and total_perm > num_perm:  
        warnings.warn("Grid space of current model spans {} combinations, switching to random sampling instead with {}/{}% coverage".format(total_perm, num_perm, total_perm))
        method = 'random'
    
    if (num_perm is None) and (method == 'exact'): # User warning if search space is very large
        if (total_perm > 1000):
            warnings.warn("Grid space of current model spans {} combinations, consider using random sampling instead".format(total_perm))

        all_perms = list(itertools.product(*z)) # iterate over entire grid space 
        
        for i in tqdm(range(0, total_perm)): # for each combination of params, train model
            grid_act_fn = [item for item in all_perms[i] if isinstance(item, str)]          # extract activation funcs of each layer
            grid_num_units = [item for item in all_perms[i] if isinstance(item, int)]       # extract no of dense units in each layer          
            model = Autoencoder(x_data.shape[1], grid_num_units, grid_act_fn, AE_type = AE_type)               # specify model, NB could implement on other types of DNNs here as well
            #model.compile(optimizer='adam', loss='mae')                                     # compile model
            history = model.full.fit(x_data, x_data, epochs=num_epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True, verbose=0)
            results.append([min(history.history["loss"]), min(history.history["val_loss"]), grid_num_units, grid_act_fn])
            
    elif isinstance(num_perm, int) and (method == 'random'):                    # ELSE randomly sample a fraction of the grid space  
        with tqdm(total=num_perm) as pbar:      
            while len(check_list)<num_perm:
                temp = copy.deepcopy(z)
                for sublist in temp:
                    random.shuffle(sublist)                                         # shuffle each sublist
                zz = [item[0] for item in temp]                                     # extract the elements in the first position of each sublist
                
                if zz not in check_list:
                    check_list.append(zz)                                               # add the new combination to the list of searched spaces if not previously searched
                    grid_act_fn = [item for item in zz if isinstance(item, str)]        # extract activation funcs of each layer
                    grid_num_units = [item for item in zz if isinstance(item, int)]     # extract no of dense units in each layer
                
                    model = Autoencoder(x_data.shape[1], grid_num_units, grid_act_fn, AE_type = AE_type) # specify model, NB could implement on other types of DNNs as well
                    #model.compile(optimizer='adam', loss='mae')                          # compile model
                    history = model.full.fit(x_data, x_data, epochs=num_epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True, verbose=0)
                    results.append([min(history.history["loss"]), min(history.history["val_loss"]), grid_num_units, grid_act_fn])
                    pbar.update(1)
    else:
         warnings.warn("Error running grid search, if method='random' selected, ensure 'num_perm' also defined")
         best_params = None
         return
       
            #check_list.append(([item for item in zz if isinstance(item, str)]+[item for item in zz if isinstance(item, int)]))
            #c = Counter(map(tuple,check_list))
            #dups = [k for k,v in c.items() if v>1]
    
    # now return best parameters
    loc_best =  [item[0] for item in results] # location of list with smallest lost
    loc_best = loc_best.index(min(loc_best))
    best_params = results[loc_best]
    
    return results, best_params    



def AE_anomaly_detection(autoencoder, train_data, test_data, ground_truth_locs, plt_title=None):    
    """ main function that applies autoencoder to the train and test sets and calculates performance of detector """
    
    # To find anomalies, first compute Reconstruction Error on Test data
    encoded_train = autoencoder.encoder.predict(train_data)
    decoded_train = autoencoder.decoder.predict(encoded_train)
    mse_train = np.mean(np.abs(train_data - decoded_train), axis=1)   # np.mean(np.power(train_data - decoded_train, 2), axis=1)
    # Then calculate error threshold based on RE distribution 
    error_threshold = np.percentile(mse_train, 95)    #np.median(mse_train)+3*np.std(mse_train)  # np.max(mse_train) 
    
    # Now compute the RE across the test points
    encoded_test = autoencoder.encoder.predict(test_data)
    decoded_test = autoencoder.decoder.predict(encoded_test)
    mse_test = np.mean(np.abs(test_data - decoded_test), axis=1)
    # get locs of all test points above the threshold, i.e., anomalies
    pred_new_attack_locs = np.where(mse_test>error_threshold)[0]
    
    # compute performance metrics on autoencoder based anomaly detector - NB normal behaviour is class 0, atacks/new attacks is class 1 
    y_true = np.zeros(len(test_data),)  
    y_true[ground_truth_locs] = 1
    
    y_pred = np.zeros(len(test_data),)
    y_pred[pred_new_attack_locs] = 1
      
    performance_summary = compute_performance_stats(y_true, y_pred)
    
     
    # Plot MSE distributions 
    sns.kdeplot(data=mse_train, bw_adjust=1, log_scale=True, color='blue', label='Reconstruction Error Train set') # need to use log scale here as dist contains very high outliers
    plt.vlines(error_threshold, ymin=0, ymax=1.0, colors='red', label='Error Threshold')
    sns.kdeplot(data=mse_test, bw_adjust=1, log_scale=True, color='green', label='Reconstruction Error Test set')
    sns.kdeplot(data=mse_test[ground_truth_locs], bw_adjust=1, log_scale=True, color='orange', label='Ground-Truth Anomalies')
    plt.xlabel("MSE")
    plt.legend(loc='upper right')
    plt.title(plt_title)
    plt.show()
    
    #save the data in a dictionary
    #results = {}
    #results['performance_summary'] = performance_summary
    #results['pred_new_attack_locs'] = pred_new_attack_locs
    #results['error_threshold'] = error_threshold
    
    return performance_summary, pred_new_attack_locs, error_threshold


def compute_performance_stats(y_true, y_pred):  
    """
        simple function to compute the performance metrics of the NIDS
    """
    # here, normal behaviour is encoded as class 0, anomaly/attack is encoded as class 1
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = conf_matrix.ravel() # where tp-> predicting a shift when indeed a shift occurs, fp-> predicting no shift(y_pred=1), when indded no shift occurs
    
    TPR = tp/(tp+fn) # true positive rate / recall
    TNR = tn/(tn+fp) # true negative rate / specificity
    PREC =  tp/(tp+fp) # precision - how often a predicted anomaly really is an anomaly
    FNR = fn/(fn+tp) # false negative rate / miss rate
    FPR = fp/(fp+tn) # false positive rate
    TS = tp/(tp+fn+fp) # threat score / critical success rate
    ACC = (tp+tn)/(tp+fp+tn+fn) # accuracy
    F1 = 2 * (PREC * TPR) / (PREC + TPR)
    
     
    return pd.DataFrame(np.array([[tn,fp,fn,tp,ACC,TPR,PREC,F1,FPR,TNR,FNR,TS]]), columns=['TN','FP','FN','TP','ACCURACY','RECALL','PRECISION','F1','FPR','TNR','FNR','T-SCORE'])


 


























# EoF
