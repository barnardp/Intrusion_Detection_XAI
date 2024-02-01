# -*- coding: utf-8 -*-
"""

Utility functions for KDD Intrusion (anomaly detection & XAI)

NB Used in XAI General directory
"""

import numpy as np
import pandas as pd

import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras import layers, Input, Model
from tensorflow import keras


import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit

import random
import itertools
import warnings
import copy
from sklearn import metrics
from os import listdir

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

    category_map = {}
    category_map[feature_names.index('protocol_type')] = list(X_train['protocol_type'].unique())
    category_map[feature_names.index('service')] = list(X_train['service'].unique())
    category_map[feature_names.index('flag')] = list(X_train['flag'].unique())
    
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
    data['category_map'] = category_map
    
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


def create_bin_model(X_train, Y_train_bin, parameters=None): 
    """ 
        Module to train binary XGBoost model using grid search
    """
    
    from sklearn.model_selection import GridSearchCV
    
    """parameters = {                                   # define the grid search parameters
        "max_depth": [3, 4, 5, 7, 9],                   # max depth of each decision tree, typically 1-10
        "learning_rate": [0.2, 0.1, 0.01, 0.05],        # typically 0.01 - 0.2
        "gamma": [0, 0.25, 0.5, 1],                     # Gamma reguliser, typically 0.0 - 0.5
        "reg_lambda": [0, 1, 5, 10],                    # lambda reguliser, typically 0 - 1
        "scale_pos_weight": [1, 3, 5],
        "subsample": [0.5, 0.7, 0.9],                   # Fract[training set] to train each tree, if too low->underfit, too high->overfit, typically 0.5-0.9
        "colsample_bytree": [0.5, 0.7, 0.9],            # fract[features] to train each tree
    }"""
    
    if parameters is None:
        parameters = {                          # define the grid search parameters
            "max_depth": [3, 5, 7, 9],          # max depth of each decision tree, typically 1-10
            "learning_rate": [0.2, 0.1, 0.01],  # typically 0.01 - 0.2           
            "subsample": [0.5, 0.7, 0.9],       # Fract[training set] to train each tree, if too low->underfit, too high->overfit, typically 0.5-0.9
            "colsample_bytree": [0.5, 0.7, 0.9],            # fract[features] to train each tree
        }
    
    model_bin = xgboost.XGBClassifier(use_label_encoder=False, objective="binary:logistic", seed= 1) 
    grid_cv = GridSearchCV(model_bin, parameters, n_jobs=-1, cv=3, scoring="roc_auc", verbose=3, refit=True)   # init grid search
    
    t_1 = time.time()
    clf = grid_cv.fit(X_train, Y_train_bin)                                               # fit models in the grid and return best model
    t_2 = time.time() - t_1
    print('Grid search took {} minutes to complete'.format( round(t_2/60) ))
    
    #best_params_bin = grid_cv.best_params_                                                  # check best parameters
    # model_bin = xgboost.XGBClassifier(**grid_cv.best_params_, objective="binary:logistic")  # OPTIONAL, can retrain final model with these parameters but using other loss fn
    # model_bin.fit(X_train, Y_train_bin) 

    return grid_cv, clf
    
class Autoencoder(Model):
    """ class to create an autonecoder
    num_units:    		numpy array with elements representing num of dense units for each layer
    act_fn:       		list with each element corresponding to the activation function of that layer
    		AE_type     		use 'joint' to train both encoder and decoder, or 'random' to keep encoder un-trained
    		num_encoder_layers 	number of layers in the encoder, ie remaining layers are for the decoder  
    """      
    
    def __init__(self, x_shape, num_units, act_fn, num_encoder_layers=3, AE_type='joint', learning_rate=1e-3):
        super(Autoencoder, self).__init__()
        self.x_shape = x_shape
        self.num_units = num_units
        self.act_fn = act_fn
        self.num_encoder_layers = num_encoder_layers # NB the last layer of the encoder corresponds to the latent layer
        self.AE_type = AE_type
        #self.mae = tf.keras.losses.MeanAbsoluteError()
        
        tf.random.set_seed(10)
        np.random.seed(10)
       
        # define the encoder
        encoder_input = Input(shape=(self.x_shape,))
        layer_i = tf.keras.layers.Dense(self.num_units[0], activation=self.act_fn[0])(encoder_input)    # add the first hidden layer
        layer_i = tf.keras.layers.Dropout(0.2)(layer_i)                                                 # add dropout to the first hidden layer
        for i in range(1, self.num_encoder_layers):
            layer_i = tf.keras.layers.Dense(self.num_units[i], activation=self.act_fn[i])(layer_i)      # add the next hidden layer  
        # finalise the encoder
        self.encoder = Model(encoder_input, layer_i, name='encoder')
        	
        # define the decoder  - NB the last layer of the decoder corresponds to the final output of the AE
        decoder_input = Input(shape=(self.num_units[i],))
        layer_i = tf.keras.layers.Dense(self.num_units[i+1], activation=self.act_fn[i+1])(decoder_input)    # add the first hidden layer
        layer_i = tf.keras.layers.Dropout(0.2)(layer_i)                                                     # add dropout to the first hidden layer
        for i in range(i+2, len(self.num_units)):
            layer_i = tf.keras.layers.Dense(self.num_units[i], activation=self.act_fn[i])(layer_i)          # add the next hidden layer 
        # finalise the decoder
        self.decoder = Model(decoder_input, layer_i, name='decoder')
        		
        		
        # now combine together into a single AE
        self.final_output = self.decoder(self.encoder(encoder_input))
        self.full = Model(encoder_input, self.final_output, name='full_AE')
        
        if AE_type == 'random':
            self.encoder.trainable = False # if using a randomised encoder, dont optimise the encoder weights
        
        # define the loss function
        #reconstruction_loss = self.mae(inputs, self.final_output)
        #self.full.add_loss(reconstruction_loss)
        self.full.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae')
        
    def call(self, x, **kwargs):
        return self.full(x)





    
class Autoencoder_2(Model):
    """ class to create an autonecoder
        num_units:    numpy array with elements representing num of dense units for each layer
        act_fn:       list with each element corresponding to the activation function of that layer
    """      
    def __init__(self, x_shape, num_units_encoder, num_units_decoder, act_fns=None, AE_type='joint', learning_rate=1e-3):
        super(Autoencoder_2, self).__init__()
        self.x_shape = x_shape
        self.num_units_encoder = num_units_encoder
        self.num_units_decoder = num_units_decoder
        self.act_fns = act_fns
        #self.mae = tf.keras.losses.MeanAbsoluteError()
        
        tf.random.set_seed(10)
        np.random.seed(10)
        
        # define the encoder
        encoder_input = Input(shape=(self.x_shape,))
        
        model = tf.keras.Sequential()
        count = 1
        for i in self.num_units_encoder:
            model.add(layers.Dense(i, activation="relu"))
            if count == 1: 
                model.add(layers.Dropout(0.2))  # add dropout to the first layer, NB in future versions, we can pass a list of layers that need dropout
            count = count + 1
        encoder_front_end = model(encoder_input)
                
        # finalise the encoder
        self.encoder = Model(encoder_input, encoder_front_end, name='encoder')
        
        # define the decoder 
        decoder_input = Input(shape=(i,))
        
        model = tf.keras.Sequential()
        count = 1
        for i in self.num_units_decoder:
            model.add(layers.Dense(i, activation="relu"))
            if count == 1: 
                model.add(layers.Dropout(0.2))  # add dropout to the first layer, NB in future versions, we can pass a list of layers that need dropout
            count = count + 1
        model.add(layers.Dense(self.x_shape, activation="tanh"))
        decoder_front_end = model(decoder_input)
        
        
        # finalise the decoder
        self.decoder = Model(decoder_input, decoder_front_end, name='decoder')
        
        # now combine together into a single AE
        self.final_output = self.decoder(self.encoder(encoder_input))
        self.full = Model(encoder_input, self.final_output, name='full_AE')
        
        if AE_type == 'random':
            self.encoder.trainable = False # if using a randomised encoder, dont optimise the encoder weights
        
        # define the loss function
        #reconstruction_loss = self.mae(inputs, self.final_output)
        #self.full.add_loss(reconstruction_loss)
        self.full.compile(optimizer='adam', loss='mae')

def get_hyper_Autoencoder(parameters, x_data, val_data=None, method='exact', num_perm=None, num_encoder_layers=3, num_epochs=20, batch_size=512, AE_type = 'joint', loss_monitor='train'):
    """
        Method to perform grid search on an autoencoder:
        parameters:         Dictionary with search vectors of each layer (i) in the network, accepts 'dense_i_units', and 'dense_i_activation' similar to Sklearn 
        x_data:             Training dataset (numpy)
        val_data:           Validation dataset, if None, method forms this set from the training data
        method:             If 'exact', the entire grid space is searched UNLESS the sapce exceeds 'num_perm'. If 'random' samples 'num_perm' searches from space
        num_perm:           Can be used to set an upper bound of searches to perform, If total grid space exceeds this value, random sampling is 
                            performed automatically. OPTIONALLY, can be set to a fractional value (0,1) to search a fraction of the total grid space
        num_encoder_layers  Number of layers in the encoder, ie used to split the 'paramters' variable into the encoder/decoder parts
        num_epochs:         The amount of epochs to train each model during the search space
        batch_size:         Batch size to use during each fit
        AE_type:            Specifies how the encoder and decoder are to be trained, use 'joint' if jointly trained, or 'random' for randomised/untrained encoder
        loss_monitor:       Which loss to examine when finding best paramters, can be 'train' or 'val'
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
            model = Autoencoder(x_data.shape[1], grid_num_units, grid_act_fn, AE_type = AE_type, num_encoder_layers = num_encoder_layers)               # specify model, NB could implement on other types of DNNs here as well
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
                
                    model = Autoencoder(x_data.shape[1], grid_num_units, grid_act_fn, AE_type = AE_type, num_encoder_layers = num_encoder_layers) # specify model, NB could implement on other types of DNNs as well
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
    if loss_monitor == 'train':
        idx = 0
    else:
        idx = 1
    loc_best =  [item[idx] for item in results] # location of list with smallest lost
    loc_best = loc_best.index(min(loc_best))
    best_params = results[loc_best]
    
    return results, best_params    


def get_hyper_Autoencoder_2(parameters, x_data, val_data=None, num_epochs=5, batch_size=512, AE_type = 'joint', max_encoder_dim=5000, min_latent_dim=8, max_decoder_dim=1000):
    """
        Method to perform ping-pong type grid search on an autoencoder:
        parameters:         Dictionary with search vectors of each layer (i) in the network, accepts 'dense_i_units', and 'dense_i_activation' similar to Sklearn 
        x_data:             Training dataset (numpy)
        val_data:           Validation dataset, if None, method forms this set from the training data
        method:             If 'exact', the entire grid space is searched UNLESS the sapce exceeds 'num_perm'. If 'random' samples 'num_perm' searches from space
        num_perm:           Can be used to set an upper bound of searches to perform, If total grid space exceeds this value, random sampling is 
                            performed automatically. OPTIONALLY, can be set to a fractional value (0,1) to search a fraction of the total grid space
        num_epochs:         The amount of epochs to train each model during the search space
        batch_size:         Batch size to use during each fit
        AE_type:            Specifies how the encoder and decoder are to be trained, use 'joint' if jointly trained, or 'random' for randomised/untrained encoder
        max_encoder_dim     maximum dimension allowed for the encoder
        min_latent_dim:     minimum dimension allowed for the latent layer
        max_decoder_dim     maximum dimension allowed for the decoder
        
    """    
    
    random.seed(1)
    tf.random.set_seed(1)
    
    best_params = {}
    results = [] # store the loss and associated model structure at each iteration    
    
    num_units_encoder = parameters['encoder_num_units']
    num_units_decoder = parameters['decoder_num_units']
    
    if val_data is None:
        x_data, val_data = train_test_split(x_data, test_size=0.2, random_state=1)
        
        
    # add more layers to the AE until the gain in performance is minimal, NB here we iterate over all 4 options and take the best performing one
    # ie model_1 -> add layer to start of encoder,      model_2 -> add layer to end of encoder (latent layer)
    #    model_3 -> add layer to end of decoder,        model_3 -> add layer to start of decoder (layer after latent layer)
    
    rel_change_in_loss = 999
    prev_loss = 999
    
    while rel_change_in_loss > 1.0: 
    
        # create temporary copies to store the current (most optimal) number of layers across the AE
        num_units_encoder_1 = num_units_encoder.copy()
        num_units_encoder_2 = num_units_encoder.copy()
        num_units_decoder_3 = num_units_decoder.copy()
        num_units_decoder_4 = num_units_decoder.copy()
        
        curr_loss = np.ones((4,1))*999
                
        if num_units_encoder[0]*2 < max_encoder_dim: # if the new layer doesnt go above the limit previously set
            num_units_encoder_1.insert(0, int(num_units_encoder[0]*2)) # insert new value at start of list, NB new value chosen to be 2 x current value at start
            model_1 = Autoencoder_2(x_data.shape[1], num_units_encoder_1, num_units_decoder, AE_type = 'joint') # train model using this configuration
            # train model and record loss history (for train and validation set)
            history_1 = model_1.full.fit(x_data, x_data, epochs=num_epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True, verbose=0)
            curr_loss[0] = min(history_1.history["val_loss"]) # record the minimum loss for this configuration
        
        if num_units_encoder[-1]/2 >= min_latent_dim: # if the new layer doesnt go below the limit previously set
            num_units_encoder_2.append(int(num_units_encoder[-1]/2)) # insert new value at end of list, NB new value chosen to be 1/2 of current value at end
            model_2 = Autoencoder_2(x_data.shape[1], num_units_encoder_2, num_units_decoder, AE_type = 'joint') # train model using this configuration
            # train model and record loss history (for train and validation set)
            history_2 = model_2.full.fit(x_data, x_data, epochs=num_epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True, verbose=0)
            curr_loss[1] = min(history_2.history["val_loss"]) # record the minimum loss for this configuration
            
        if num_units_decoder[-1]*2 < max_decoder_dim: # if the new layer doesnt go above the limit previously set
            num_units_decoder_3.append(int(num_units_decoder[-1]*2)) # insert new value at start of list, NB new value chosen to be 2 x current value at start
            model_3 = Autoencoder_2(x_data.shape[1], num_units_encoder, num_units_decoder_3, AE_type = 'joint') # train model using this configuration
            # train model and record loss history (for train and validation set)
            history_3 = model_3.full.fit(x_data, x_data, epochs=num_epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True, verbose=0)
            curr_loss[2] = min(history_3.history["val_loss"]) # record the minimum loss for this configuration
        
        if num_units_decoder[0]/2 >= min_latent_dim: # if the new layer doesnt go below the limit previously set
            num_units_decoder_4.insert(0, int(num_units_decoder[0]/2)) # insert new value at end of list, NB new value chosen to be 1/2 of current value at end
            model_4 = Autoencoder_2(x_data.shape[1], num_units_encoder, num_units_decoder_4, AE_type = 'joint') # train model using this configuration
            # train model and record loss history (for train and validation set)
            history_4 = model_4.full.fit(x_data, x_data, epochs=num_epochs, batch_size=batch_size, validation_data=(val_data, val_data), shuffle=True, verbose=0)
            curr_loss[3] = min(history_4.history["val_loss"]) # record the minimum loss for this configuration    
                
            
        rel_change_in_loss = prev_loss/curr_loss.min()  # calculate the relative change in loss
        prev_loss = curr_loss.min()                     # update the new best loss obtained
        
        if rel_change_in_loss > 1.0:
            # update encoder or decoder based on the option that achieved new lowest error
            if curr_loss.argmin() == 0:
                num_units_encoder = num_units_encoder_1.copy()
            elif curr_loss.argmin() == 1:
                num_units_encoder = num_units_encoder_2.copy()
            elif curr_loss.argmin() == 2:
                num_units_decoder = num_units_decoder_3.copy()
            elif curr_loss.argmin() == 3:
                num_units_decoder = num_units_decoder_4.copy()
            
        results.append([num_units_encoder, num_units_decoder, curr_loss.min()])
        print('.')
         
        # add the best parameters into a dictionary
        temp = [item[2] for item in results]
        temp = results[ temp.index(min(temp))  ]
        best_params['encoder_num_units'] = temp[0]
        best_params['decoder_num_units'] = temp[1]
        
    return results, best_params


def second_stage(attrib_train, attrib_test, xai_name, X_test, Y_test_true, Y_prob_test, new_attack_locs):
    """ function that implements the second stage of our anomaly detector, ie given the explanations, train an autoencoder and then compute performance metrics
        
        attrib_train:       Explanations from the train set
        attrib_test:        Explanations from the test set
        xai_name:           Name (string) of the explanation method used, ie "SHAP", "lime" etc
        X_test:             Test set data
        Y_test_true:        Test set ground truth labels
        Y_prob_test:        Probability predictions (nsamples, 2) from the first stage of the pipeline, ie the XGBoost Model
    """
    results = {}
    
    results[xai_name + '_train'] = attrib_train
    results[xai_name + '_test'] = attrib_test
    
    # scale data using Sklearn minmax scaler
    results['scaler'] = MinMaxScaler(feature_range=(-1,1))                                     
    results[xai_name + '_train_scaled'] = results['scaler'].fit_transform(attrib_train)  # scale the training set data
    results[xai_name + '_test_scaled'] = results['scaler'].transform(attrib_test)        # scale the test set data
    
    # before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
    x_data, val_data = train_test_split(results[xai_name + '_train_scaled'], test_size=0.2, random_state=10)
    
    results['best_params'] = [100, 100, [1456, 724, 8, 632, 1644, 41], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]   
          
    # Using the best parameters, build and train the final model
    AE_model = Autoencoder(x_data.shape[1], results['best_params'][2], results['best_params'][3], AE_type='joint') # create the AE model object
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=40, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
    results['history'] = AE_model.full.fit(x_data, x_data, epochs=1000, batch_size=512, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
    # plot the training curve
    plt.plot(results['history']["loss"], label="Training Loss")
    plt.plot(results['history']["val_loss"], label="Validation Loss")
    plt.title("Training curve for " + xai_name)
    plt.xlabel("No. of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # perform anomaly detection based on the reconstruction error of the AE and save results
    results['performance_new_attacks'], results['new_attack_pred_locs'], results['AE_threshold'] = AE_anomaly_detection(AE_model, results[xai_name + '_train_scaled'], results[xai_name + '_test_scaled'], new_attack_locs, plt_title=('Autoencoder loss for ' + xai_name + ' based explanations'))
    
    # calculate overall accuracy of the IDS system (XGBoost IDS and Anomaly detector) to detect attacks new or old attacks on the NSL-KDD Testset+
    # NB To compute the ROC and AUC, we need to know the probability of of our overall classifier for class 1 (Attack). Here, we use the proabilities from the initial xgboost model, but replace any of the new attack samples with probability 1.0
    y_prob = Y_prob_test[:,1].copy()
    y_prob[ results['new_attack_pred_locs'] ] = 1.0
    results['all_attack_pred_locs'] = np.where(y_prob > 0.5)[0]
    results['y_pred_all'] = np.zeros(len(Y_test_true),)
    results['y_pred_all'][results['all_attack_pred_locs']] = 1
 
    results['performance_overall'] = compute_performance_stats(Y_test_true, results['y_pred_all'], y_prob)
    
    return results, AE_model

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
    
    #save the data in a dictionary
    #results = {}
    #results['performance_summary'] = performance_summary
    #results['pred_new_attack_locs'] = pred_new_attack_locs
    #results['error_threshold'] = error_threshold
    
    return performance_summary, pred_new_attack_locs, error_threshold




def compute_performance_stats(y_true, y_pred, y_prob=None, title=None, rnd=4):  
    """
        simple function to compute the performance metrics of the NIDS
    """
    # here, normal behaviour is encoded as class 0, anomaly/attack is encoded as class 1
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    
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


 


""" *************************** Utility file for Miel's CICIDS17/18 evaluation *************************** """

def generate_clean_data(data_loc, new_data_loc):
    """
        Original file has typo where Tuesday data is called 'Thuesday'
        The 'Thuesday' set is the only one that has ALL 84 features in it (inc. the label column). The extra features in this set BUT not in the others include: Flow ID, Src IP, Src Port, Dst IP
        
        NB this file is adapted directly from the original authors code: https://gitlab.ilabt.imec.be/mverkerk/cic-ids-2018/-/blob/master/notebooks/generate_cleaned_data.py
        
    """
   
    files = listdir(data_loc) # get name of all files in the dataset directory
    
    for day, filename in enumerate(files):
        print(f"------- {filename} -------")
        df = pd.read_csv(f"{data_loc}{filename}", skipinitialspace=True)
    #     print(df["Label"].value_counts())
    #     print(df.columns[df.dtypes == "object"])
        
        print(f"shape: {df.shape}")
        # Drop destination port?  "Dst Port"
        df.drop(columns=["Flow ID", "Src IP", "Src Port", "Dst IP", 
                         'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
                           'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
                           'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'], inplace=True, errors="ignore")
        
        # Drop rows with invalid data
        cols=[i for i in df.columns if i not in ["Timestamp", "Label"]]
        for col in cols:
            df[col]=pd.to_numeric(df[col], errors='coerce')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
    #     df = df.apply(pd.to_numeric, errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"{df.isna().any(axis=1).sum()} rows dropped")
        df.dropna(inplace=True)
        print(f"shape: {df.shape}")
        
        # Drop duplicate rows
        df.drop_duplicates(inplace=True, subset=df.columns.difference(['Label', 'Timestamp']))
        print(f"shape: {df.shape}")
        
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x + pd.Timedelta(hours=12) if x.hour < 8 else x)
        df = df.sort_values(by=['Timestamp'])
        df[df["Timestamp"] > "2018-01-01"].to_csv(f"{new_data_loc}/{filename}", index=False)

def GroundTruthGeneration(DATA_DIR):
    """
        
        Function adapted from: https://gitlab.ilabt.imec.be/mverkerk/cic-ids-2018/-/blob/master/notebooks/GroundTruthGeneration.ipynb
        NB. The original code performs numerous visualisation steps that helps to see the shapes of each class etc. Here we comment these
        parts out for time efficiency.

        DATA_DIR:       Full location to where the previously computed 'clean' data is stored, in our case DATA_DIR = new_data_loc_18
    """
    
    files = [
        "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    ]
    
    all_data = pd.DataFrame()
    
    for day, filename in enumerate(files):
        print(f"***** {filename} *****")
        df = pd.read_csv(f"{DATA_DIR}{filename}", skipinitialspace=True, parse_dates=["Timestamp"])
        print(df["Label"].value_counts())
        print()
        all_data = all_data.append(df, ignore_index=True)
    print("***** TOTAL *****")
    all_data["Label"].value_counts()
    
    
    all_data.to_csv(f"{DATA_DIR}all_data.csv", index=False) # this can take over 30 mins to complete
    all_malicious = all_data[all_data["Label"] != 'Benign']
    all_malicious.drop_duplicates(inplace=True, subset=all_malicious.columns.difference(['Label', 'Timestamp']))
    all_malicious.Label.value_counts()
    
    all_malicious.shape
    all_malicious.to_csv(f"{DATA_DIR}all_malicious.csv", index=False)
    
    all_benign = all_data[all_data["Label"] == 'Benign']
    all_benign.drop_duplicates(inplace=True, subset=all_benign.columns.difference(['Label', 'Timestamp']))
    all_benign.shape
    
    
    benign_1M = all_benign.sample(n=1000000, random_state=0)
    benign_1M.shape
    
    benign_1M.to_csv(f"{DATA_DIR}benign_1M.csv", index=False)
    all_benign.to_csv(f"{DATA_DIR}all_benign.csv", index=False)
    
    
    malicious_1M, malicious_rest = train_test_split(all_malicious, train_size=1000000, random_state=42, stratify=all_malicious.Label, shuffle=True)
    malicious_1M.Label.value_counts()
    malicious_rest.Label.value_counts()
     
    df = pd.read_csv(f"{DATA_DIR}all_malicious.csv")
    df['Label'].value_counts()
    df.shape
    
    df['type'] = df['Label'].map(
        {
            'Benign':'Benign', 
            'DDoS attacks-LOIC-HTTP': '(D)DOS',
            'DDOS attack-HOIC': '(D)DOS',
            'DoS attacks-Hulk': '(D)DOS',
            'Bot': 'Botnet',
            'Infilteration': 'Infiltration',
            'SSH-Bruteforce': 'Brute Force',
            'DoS attacks-GoldenEye': '(D)DOS',
            'DoS attacks-Slowloris': '(D)DOS',
            'DDOS attack-LOIC-UDP': '(D)DOS',
            'Brute Force -Web': 'Web Attack',
            'Brute Force -XSS': 'Web Attack',
            'SQL Injection': 'Web Attack',
            'FTP-BruteForce': 'Brute Force',
            'DoS attacks-SlowHTTPTest': '(D)DOS'
        })
    df['type'].value_counts()
    
    df['Label'].value_counts() / 8982289


# Use same code from authors, with exeption of changing labels for benign = 0 and attack = 1 and callin the
# load_data_fraud function inside this one so that all variables can be directly returned inside a dictionary
def load_data_cicids_miels(data_dir, filename, verbose=True, train_size=50000, test_size=None, validation_perc=None):
    
    df = pd.read_csv(f"{data_dir}/{filename}")
    Y = df["Label"].map(lambda x: 0 if (x == "Benign") else 1)
    labels = df["Label"]
    df.drop(columns=["Label", "Timestamp", "Dst Port"], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, Y, train_size=train_size, test_size=test_size, shuffle=True, stratify=Y, random_state=42)
    if verbose:
        print("***** Train Data *****")
        print(labels.loc[y_train.index].value_counts())
        print("***** Test Data *****")
        print(labels.loc[y_test.index].value_counts())
            
    train_labels = labels.loc[y_train.index]
    test_labels = labels.loc[y_test.index]
    
    # for simplicity, compute the fraud samples here instead of the main code
    X_fraud, y_fraud, labels_fraud = load_data_fraud_miels(data_dir, verbose=False)
    print("Total number of attacks in datatset = " + str(len(y_fraud)) + "\n")
    # Concat benign transaction to fraud transactions for evaluation
    X_test = X_test.append(X_fraud)
    y_test = y_test.append(y_fraud)
    test_labels = test_labels.append(labels_fraud)
    
    # split into validation and test 
    X_val, X_t, y_val, y_t, label_val, label_t = train_test_split(X_test, y_test, test_labels, train_size=validation_perc, random_state=42, stratify=test_labels, shuffle=True)
    
    # As final step, place all vairables into a single dictionary
    data = {}
    data['x_train'] = X_train
    data['x_validation'] = X_val
    data['x_test'] = X_t
    data['y_train'] = y_train
    data['y_validation'] = y_val
    data['y_test'] = y_t
    data['label_train'] = train_labels
    data['label_validation'] = label_val
    data['label_test'] = label_t
    
    return data


def load_data_fraud_miels(data_dir, verbose=True):
    df = pd.read_csv(f"{data_dir}/all_malicious.csv")
    Y = df["Label"].map(lambda x: 0 if (x == "Benign") else 1)
    labels = df["Label"]
    df.drop(columns=["Label", "Timestamp", "Dst Port"], inplace=True)
    if verbose:
        print("***** Data *****")
        print(labels.value_counts())
    return df, Y, labels

def clean_dataset(data_loc, new_data_loc):
    """
        - Original CICIDS2018 datatset(s) have a typo where the Tuesday file is called 'Thuesday'
        - The 'Thuesday' set is the only one that has ALL 84 features in it (inc. the label column). The extra features found in this set BUT not in the others include: Flow ID, Src IP, Src Port, Dst IP
        - The CICIDS2017 datatset has upto 85 features. Closer investigation finds that the the "Fwd Header Length" is repeated twice, the second called 'Fwd Header Length.1'
        
        - NB this file is adapted directly from the original authors code: https://gitlab.ilabt.imec.be/mverkerk/cic-ids-2018/-/blob/master/notebooks/generate_cleaned_data.py
        
    """
   # dictionary that maps the features in the CICIDS2017 to CICIDS2018
    mapper =    {      # CICIDS2017 : CICIDS2018
                        'Source IP': 'Src IP',
                        'Source Port': 'Src Port',
                        'Destination IP': 'Dst IP',
                        'Destination Port': 'Dst Port',
                        'Total Fwd Packets': 'Tot Fwd Pkts',
                        'Total Backward Packets': 'Tot Bwd Pkts',
                        'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
                        'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
                        'Fwd Packet Length Max': 'Fwd Pkt Len Max',
                        'Fwd Packet Length Min': 'Fwd Pkt Len Min',
                        'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
                        'Fwd Packet Length Std': 'Fwd Pkt Len Std',
                        'Bwd Packet Length Max': 'Bwd Pkt Len Max',
                        'Bwd Packet Length Min': 'Bwd Pkt Len Min',
                        'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
                        'Bwd Packet Length Std': 'Bwd Pkt Len Std',
                        'Flow Bytes/s': 'Flow Byts/s',
                        'Flow Packets/s': 'Flow Pkts/s',
                        'Fwd IAT Total': 'Fwd IAT Tot',
                        'Bwd IAT Total': 'Bwd IAT Tot',
                        'Fwd Header Length': 'Fwd Header Len',
                        'Bwd Header Length': 'Bwd Header Len',
                        'Fwd Packets/s': 'Fwd Pkts/s',
                        'Bwd Packets/s': 'Bwd Pkts/s',
                        'Min Packet Length': 'Pkt Len Min',
                        'Max Packet Length': 'Pkt Len Max',
                        'Packet Length Mean': 'Pkt Len Mean',
                        'Packet Length Std': 'Pkt Len Std',
                        'Packet Length Variance': 'Pkt Len Var',
                        'FIN Flag Count': 'FIN Flag Cnt',
                        'SYN Flag Count': 'SYN Flag Cnt',
                        'RST Flag Count': 'RST Flag Cnt',
                        'PSH Flag Count': 'PSH Flag Cnt',
                        'ACK Flag Count': 'ACK Flag Cnt',
                        'URG Flag Count': 'URG Flag Cnt',
                        'ECE Flag Count': 'ECE Flag Cnt',
                        'Average Packet Size': 'Pkt Size Avg',
                        'Avg Fwd Segment Size': 'Fwd Seg Size Avg',
                        'Avg Bwd Segment Size': 'Bwd Seg Size Avg',
                        'Fwd Avg Bytes/Bulk': 'Fwd Byts/b Avg',
                        'Fwd Avg Packets/Bulk': 'Fwd Pkts/b Avg',
                        'Fwd Avg Bulk Rate': 'Fwd Blk Rate Avg',
                        'Bwd Avg Bytes/Bulk': 'Bwd Byts/b Avg',
                        'Bwd Avg Packets/Bulk': 'Bwd Pkts/b Avg',
                        'Bwd Avg Bulk Rate': 'Bwd Blk Rate Avg',
                        'Subflow Fwd Packets': 'Subflow Fwd Pkts',
                        'Subflow Fwd Bytes': 'Subflow Fwd Byts',
                        'Subflow Bwd Packets': 'Subflow Bwd Pkts',
                        'Subflow Bwd Bytes': 'Subflow Bwd Byts',
                        'Init_Win_bytes_forward': 'Init Fwd Win Byts',
                        'Init_Win_bytes_backward': 'Init Bwd Win Byts',
                        'act_data_pkt_fwd': 'Fwd Act Data Pkts',
                        'min_seg_size_forward': 'Fwd Seg Size Min'
                         
                }
    
    
    files = [
                 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                 'Monday-WorkingHours.pcap_ISCX.csv',
                 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                 'Tuesday-WorkingHours.pcap_ISCX.csv',
                 'Wednesday-workingHours.pcap_ISCX.csv'] # get name of all files in the dataset directory
    
    #all_data = pd.DataFrame()
    
    for day, filename in enumerate(files):
        print(f"------- {filename} -------")
        df = pd.read_csv(f"{data_loc}{filename}", skipinitialspace=True, encoding='latin')
        #print(df["Label"].value_counts())
    #     print(df.columns[df.dtypes == "object"])
        # delete the repeated column, 'Fwd Header Length.1'
        df.drop(columns=['Fwd Header Length.1'], inplace=True)   
        # Rename column names to those used in CICIDS18 - to ensure uniformity across files
        df.rename(columns=mapper, inplace=True)
        # also the rename any benigh labels
        df['Label'][df.Label == 'BENIGN'] = 'Benign'         #   df['Label'].value_counts()
        
        #all_data = all_data.append(df, ignore_index=True)   # 3119345
              
        #df = all_data.copy()
    
        #print(f"shape: {df.shape}")
        # Drop destination port?  "Dst Port"
        df.drop(columns=["Flow ID", "Src IP", "Src Port", "Dst IP", 
                         'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count',
                           'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
                           'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'], inplace=True, errors="ignore")
        
        # Drop rows with invalid data
        cols=[i for i in df.columns if i not in ["Timestamp", "Label"]]
        for col in cols:
            df[col]=pd.to_numeric(df[col], errors='coerce')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
        # From paper, the following number of instances are supposed to be present for each class:
            # Benign                              2,271,320   (we have 2,146,984)
            # Dos 230124+128025+10293+5796+5499 = 379,737
            # Portscan                            158,804
            # Brute Force 7935+5897             = 13,832
            # Web Attack 1507+652+21            = 2,180
            # Botnet                              1,956
            # Infiltration                        36   
            # Heartbleed                          11
            #
            #                            Total = 2,827,876, Total Attacks = 556,556
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #print(f"{df.isna().any(axis=1).sum()} rows dropped")
        df.dropna(inplace=True)
        #print(f"shape: {df.shape}")  # NB after this step, the number of samples is reduced to 2,827,876...similar to the figure in the paper
        
        # Drop duplicate rows - NB If we perform this step, the number of samples drop to 2520273, imlpying that the original paper must have a typo in Fig 1
        #df.drop_duplicates(inplace=True, subset=df.columns.difference(['Label', 'Timestamp'])) # After this, the number of attacks = 425741. NB we perform this step later.
        #print(f"shape: {df.shape}")
        
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x + pd.Timedelta(hours=12) if x.hour < 8 else x)
        df = df.sort_values(by=['Timestamp'])
        df.to_csv(f"{new_data_loc}/{filename}", index=False)
        
        
        

def aggregate_data(DATA_DIR):
    """
        
        Function adapted from: https://gitlab.ilabt.imec.be/mverkerk/cic-ids-2018/-/blob/master/notebooks/GroundTruthGeneration.ipynb
        NB. The original code performs numerous visualisation steps that helps to see the shapes of each class etc. Here we comment these
        parts out for time efficiency.

        DATA_DIR:       Full location to where the previously computed 'clean' data is stored, in our case DATA_DIR = new_data_loc_18
    """


    files = [   'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                'Monday-WorkingHours.pcap_ISCX.csv',
                'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                'Tuesday-WorkingHours.pcap_ISCX.csv',
                'Wednesday-workingHours.pcap_ISCX.csv' ]
    
    all_data = pd.DataFrame()
    
    for day, filename in enumerate(files):
        print(f"***** {filename} *****")
        df = pd.read_csv(f"{DATA_DIR}{filename}", skipinitialspace=True, parse_dates=["Timestamp"])
        #print(df["Label"].value_counts())
        #print()
        all_data = all_data.append(df, ignore_index=True)
    print("***** TOTAL *****")
    all_data["Label"].value_counts()
    
    # z = np.sum(all_data["Label"].value_counts()) - all_data["Label"].value_counts()['Benign']
    
    all_data.to_csv(f"{DATA_DIR}all_data.csv", index=False) 
    all_malicious = all_data[all_data["Label"] != 'Benign']
    all_malicious.drop_duplicates(inplace=True, subset=all_malicious.columns.difference(['Label', 'Timestamp']))
    all_malicious.Label.value_counts()
    all_malicious.to_csv(f"{DATA_DIR}all_malicious.csv", index=False)
    
    all_benign = all_data[all_data["Label"] == 'Benign']
    all_benign.drop_duplicates(inplace=True, subset=all_benign.columns.difference(['Label', 'Timestamp']))
    all_benign.to_csv(f"{DATA_DIR}all_benign.csv", index=False)
      
   
""" ****************** End of Miels Utility functions ****************** """
   
   
def shap_transform_scale(shap_values, expected_value, model_prediction):
    # Function to approximately transform the SHAP values in logit form into preobability form
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


def second_stage(x_test_B, y_test_B, shap_train, shap_val, shap_test_B, trial=0):
    
    # scale data using Sklearn minmax scaler, NB here we fit the scaler to both the train and val normal data
    scaler = MinMaxScaler(feature_range=(-1,1))          
    scaler.fit( np.concatenate((shap_train, shap_val), axis=0) );                         
    shap_train_scaled = scaler.transform(shap_train)       # scale the training set data
    shap_val_scaled = scaler.transform(shap_val)           # scale the validation set data
    # shap_test_A_scaled = scaler.transform(shap_test_A)         # scale the test set data
    shap_test_B_scaled = scaler.transform(shap_test_B)         # scale the test set data
    
    # before training autoencoder, split the SHAP values (based on the training data) into a new train and validation set
    #x_data, val_data = train_test_split(shap_train_scaled, test_size=0.2, random_state=10)
    x_data = shap_train_scaled
    val_data = shap_val_scaled
    
    # perform grid search to find the best paramters to use for the autoencoder model
    # specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
    """
    parameters = {
                                    # encoder params to search across
                                    'dense_1_units':[344],                                   'dense_1_activation':['relu'],  # 32,64,128,256
                                    'dense_2_units':[128],                                 'dense_2_activation':['relu'],  # 32,48,64
                                    'dense_3_units':[19],                                           'dense_3_activation':['relu'], # 8,16, 24
                                    # decoder params to search across
                                    'dense_4_units':[344],                                        'dense_4_activation':['relu'],
                                    'dense_5_units':[1440],                                    'dense_5_activation':['relu'],  # 64,128,256
                                    'dense_6_units':[x_data.shape[1]],               'dense_6_activation':['tanh']    
                                }
    
    # check size of grid space to ensure not too large
    z = [*parameters.values()]                       # get values of each sublist in the overall parameter list 
    z = np.prod(np.array([len(sublist) for sublist in z]))              # total number of permutations in the grid 
    
    # perform the grid search and return parameters of the best model
    _, best_params = utf.get_hyper_Autoencoder(parameters, x_data, val_data, method='exact', num_epochs=10, batch_size=1024, AE_type = 'joint', loss_monitor='val')   
    
    """
    
    # best_params = [[],[],[1812, 1244, 21, 66, 1268, 67], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  # NB these params were calculated for potentially corrputed Sabaas values
    best_params = [[],[],[344, 128, 19, 344, 1440, 67], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']] 
    # Using the best parameters, build and train the final model
    AE_model = utf.Autoencoder(x_data.shape[1], best_params[2], best_params[3], learning_rate=1e-4) # create the AE model object
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
    history_AE_SHAP = AE_model.full.fit(x_data,x_data, epochs=1000, batch_size=1024, shuffle=True, validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
    # plot the training curve
    plt.plot(history_AE_SHAP["loss"], label="Training Loss")
    plt.plot(history_AE_SHAP["val_loss"], label="Validation Loss")
    plt.title("Train & Validation Loss for trial no. " + str(trial))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    #results = {}
  
    ground_truth_locs = np.where(y_test_B == 1)[0]
    # use the explanations to identify the new attack samples on the CICIDS17 Dataset (Intra)
    temp, pred_new_attack_locs, _ = utf.AE_anomaly_detection(AE_model, x_data, shap_test_B_scaled, ground_truth_locs, plt_title='Reconstruction Error for trial no. ' + str(trial), threshold=95)
    # get the initial probability predictions of the XGBoost model, isolate class 1 (Attack)
    y_prob = model.predict_proba(x_test_B)[:, 1]
    y_prob[pred_new_attack_locs] = 1.0 # update the new attacks
    y_pred = (y_prob>0.5)*1
    # compute final results
    results_overall_inter = utf.compute_performance_stats(y_test_B, y_pred, y_prob)
    
    return results_overall_inter




""" ************************************ Unused Code and functions ************************************ """


def filter_new_data_kdd(new_data, x_train):
    """ function to filter new data so that it adheres to the data types trained on the NSL KDD dataset 
        
        new_data:       the unstandardised new data from the VAE/GAN (numpy array)
        x_train:        training dataset (numpy array), used to infer typical data types/values
    """
    # nb this assumed dataset composed of label encoding scheme, if using other scheme, these indices need to be updated accordingly
    kdd_categorical_feats = np.array([1,2,3], dtype=int)
    kdd_binary_feats = np.array([6, 11, 13, 19, 20, 21], dtype=int)
    kdd_discrete_feats = np.array([7,8,14,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], dtype=int)
    # kdd_continuous_feats = np.array([0,  4,  5,  9, 10, 12, 15, 16, 17, 18], dtype=int)
    
    # for each feature, check its type:
    # If categorical/binary, round to nearest category
    # If numerical, round to nearest integer 
    
    # process categorical features, NB tested for simple label encoder
    for feat in kdd_categorical_feats:
        vals = np.unique(x_train[:,feat])                             # get a list of all the unique values this feature occupies
        temp = new_data[:,feat]                                                     # isolate the column of this feature
        new_data[:,feat] = np.digitize(np.rint(temp),bins=vals[:-1], right=True)    # write processed column back into the matrix
    
    # process binary features, NB tested for simple label encoder
    for feat in kdd_binary_feats:
        vals = np.unique(x_train[:,feat])                             # get a list of all the unique values this feature occupies
        temp = new_data[:,feat]                                                     # isolate the column of this feature
        new_data[:,feat] = np.digitize(np.rint(temp),bins=vals[:-1], right=True)    # write processed column back into the matrix
    
    # process discrete features, NB tested for simple label encoder
    for feat in kdd_discrete_feats:
        vals = np.arange(np.min(x_train[:,feat]), np.max(x_train[:,feat])+1) # create list covering entire range of unique values this feature occupies
        temp = new_data[:,feat]                                                     # isolate the column of this feature
        new_data[:,feat] = np.digitize(np.rint(temp),bins=vals[:-1], right=True)    # write processed column back into the matrix

    return new_data

def build_model(hp):
    model = keras.Sequential()
    # encoder
    model.add(keras.layers.Dense(
        hp.Choice('layer_1_units', [16, 256, 16]),
        activation=hp.Choice('activation_1', ['relu', 'tanh']),
        input_shape = (41,) ))
    model.add(keras.layers.Dense(
        hp.Choice('layer_2_units', [16, 256, 16]),
        activation=hp.Choice('activation_2', ['relu', 'tanh']) ))
    model.add(keras.layers.Dense(
        hp.Choice('layer_3_units', [8, 32, 6]),
        activation=hp.Choice('activation_3', ['relu', 'tanh']) ))
    # decoder
    model.add(keras.layers.Dense(
        hp.Choice('layer_4_units', [16, 256, 16]),
        activation=hp.Choice('activation_4', ['relu', 'tanh']) ))
    model.add(keras.layers.Dense(
        hp.Choice('layer_5_units', [16, 256, 16]),
        activation=hp.Choice('activation_5', ['relu', 'tanh']) ))
    model.add(keras.layers.Dense(41, activation='tanh'))
    
    model.compile(loss='mae')

    return model

# Can use MLJAR to get optimal params as well, https://github.com/mljar/mljar-supervised
def KT_hypermodel_AE(x_data, val_data=None, save_path=None, max_trials=25, exe_per_trial=5, epochs=15, batch_size=512, over_write=False):
    
    if val_data is None:
        x_data, val_data = train_test_split(x_data, test_size=0.2, random_state=1)
        
    if save_path is not None: # go to the specified directory
        import os
        orig_dir = os.getcwd()
        os.chdir(save_path)
        
    # setup tuner
    tuner = kt.BayesianOptimization(build_model, seed=10, objective='val_loss', max_trials=max_trials, executions_per_trial=exe_per_trial, overwrite=over_write, project_name='KTuner')
    # perform grid search
    tuner.search(x_data, x_data, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_data))   
    # Get the optimal hyperparameters
    best_params = tuner.get_best_hyperparameters()[0] # print(best_params.values)
    # Build the model with the optimal hyperparameters and train it on the data
    best_model = tuner.hypermodel.build(best_params)
    
    if save_path is not None: # return to the original directory
        os.chdir(orig_dir)
        
    return best_model, best_params
    

def KT_fit_autoencoder(x_data, val_data, params):
    """ function to train autoencoder based on params returned by Keras Tuner """
    
    np.random.seed(10)
    tf.random.set_seed(10)
    
    # parse params from keras tuner
    units = [item for item in params.values.values() if isinstance(item, int)]
    act_funcs = [item for item in params.values.values() if isinstance(item, str)]
    
    
    autoencoder = Autoencoder(x_data.shape[1], units, act_funcs)
    #autoencoder.compile(optimizer='adam', loss='mae') # loss = 'mae'
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, 
                                                  patience=25, verbose=2, mode='min', restore_best_weights=True)     # define our early stopping
    history = autoencoder.fit(x_data, x_data, epochs=1000, batch_size=512, shuffle=True,
                              validation_data=(val_data, val_data), verbose=2, callbacks=[early_stop]).history
    
    
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # tensorflow cant save composite autencoder object, so will need to return each encoder/decoder seperately
    # however, to allow possibility of transfer learning both models later, we also need to re-compile them
    #autoencoder.encoder.compile(optimizer='adam', loss='mae')
    #autoencoder.decoder.compile(optimizer='adam', loss='mae')

    return autoencoder, history

    
    
    
    
    
    
    
    
    
    
    
    
# EoF