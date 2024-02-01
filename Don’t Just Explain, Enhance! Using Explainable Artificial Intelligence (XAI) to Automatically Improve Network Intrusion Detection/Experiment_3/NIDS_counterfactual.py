

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

from alibi.explainers import CounterfactualRL

from xgboost import XGBClassifier
from tensorflow.keras import Input, Model
from sklearn import metrics
import pickle
import time
import os
os.chdir('D:/')              # location where files are stored (main + utility)
save_loc = 'D:/'    # location to save results
   
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
        self.full.compile(optimizer='adam', loss='mae')
        
    def call(self, x: tf.Tensor, training: bool = True, **kwargs) -> tf.Tensor:
        return self.full(x)
    
    
# Load  dataset.
data_kdd = pickle.load(open(save_loc + "data_nsl.pkl", "rb"))

X_train = data_kdd['X_train'].values
X_test = data_kdd['X_test'].values

# construct and train the classifier
model = XGBClassifier(use_label_encoder=False, objective="binary:logistic", seed= 10) 
model.fit(X_train, data_kdd['Y_train_bin'])
    
# Define prediction function
predictor = lambda x: model.predict_proba(x)
# y_pred = predictor(X_test) 
   
# Load pre-trained autoencoder for NSL-KDD Dataset
AE_params = [[1400, 250, 6, 200, 200, 41], ['relu', 'relu', 'relu', 'relu', 'relu', 'tanh']]  
AE_model = Autoencoder(41, AE_params[0], AE_params[1]) 
AE_model.full.load_weights(save_loc + 'AE_model_normal_weights'); 
# y_pred = AE_model(X_test)

# Define constants
LATENT_DIM = 6
COEFF_SPARSITY = 0.5                # sparisty coefficient
COEFF_CONSISTENCY = 0.5             # consisteny coefficient -> no consistency
TRAIN_STEPS = 50000                  # number of training steps -> consider increasing the number of steps
BATCH_SIZE = 1000                   # batch size    
    
# Define explainer.
explainer = CounterfactualRL(predictor=predictor,
                             encoder=AE_model.encoder,
                             decoder=AE_model.decoder,
                             latent_dim=LATENT_DIM,
                             coeff_sparsity=COEFF_SPARSITY,
                             coeff_consistency=COEFF_CONSISTENCY,
                             train_steps=TRAIN_STEPS,
                             batch_size=BATCH_SIZE,
                             backend="tensorflow")    
    
    
# Fit the explainer
explainer = explainer.fit(X=X_train[:1000])    # NB train and test data passed to the explainer need to be numpy arrays, DataFrame will cause error
    
exp_cfl = {}  
    
# Generate counterfactuals for the test set instances
y_pred = model.predict(X_test)
target_labels = (y_pred == 0)*1   # flip the predicted test set classes so that the explanations explain how to get to the other class

dt = time.time()
explanations_obj = explainer.explain(X_test, Y_t=target_labels, batch_size=1000)    
dt = (time.time() - dt)/len(X_test)
    
exp_cfl['exp_test'] = explanations_obj.cf['X']
exp_class = explanations_obj.cf['class'].reshape(-1,)   # equal to model.predict(exp_cfl['exp_test'])
    
# check how many of the target labels we actually achieved
np.sum(target_labels == exp_class)/len(target_labels)  # on the test set we get a success rate of 95.9%
  
# Generate counterfactuals for the train set instances
y_pred = model.predict(X_train)
target_labels = (y_pred == 0)*1   # flip the predicted test set classes so that the explanations explain how to get to the other class
explanations_obj = explainer.explain(X_train, Y_t=target_labels, batch_size=1000)        
exp_cfl['exp_train'] = explanations_obj.cf['X']
exp_class = explanations_obj.cf['class'].reshape(-1,)   # equal to model.predict(exp_cfl['exp_test'])
# check how many of the target labels we actually achieved
np.sum(target_labels == exp_class)/len(target_labels)  # on the test set we get a success rate of 98.9%
      
# now save the explanations - Will be processed further in the main script
# explainer_new.params['seed']
pickle.dump(exp_cfl, open(save_loc + "exp_cfl.pkl", "wb"))     
    
# Also save the trained explainer
explainer.save(save_loc + "explainer_cfl")    
    
"""     

explainer_new = CounterfactualRL(predictor=predictor,
                             encoder=AE_model.encoder,
                             decoder=AE_model.decoder,
                             latent_dim=LATENT_DIM,
                             coeff_sparsity=COEFF_SPARSITY,
                             coeff_consistency=COEFF_CONSISTENCY,
                             train_steps=TRAIN_STEPS,
                             batch_size=BATCH_SIZE,
                             backend="tensorflow", seed=1)        
    
# NB seed doesnt include the tensorflow backend => the loaded model wont be able to exactly re-produce the original train and test set explanations
# One way that might work to reproduce results could be to load the explainer and use the results from the loaded one. If loading the explainer at a 
# later date results in the same TF seed then this would work...Further analysis shows that the success rate drops dramatically on the loaded explainer
# therefore there doesnt seem to be a reliable way to load the explainer

from alibi.saving import load_explainer

explainer_new.load(save_loc + "explainer_cfl", predictor)    
    
explanations_obj = explainer_new.explain(X_test, Y_t=target_labels, batch_size=1000)    
    
z = explanations_obj.cf['X']   


y_pred = model.predict(X_test)
target_labels = (y_pred == 0)*1   # flip the predicted test set classes so that the explanations explain how to get to the other class

explanations_obj = explainer_new.explain(X_test, Y_t=target_labels, batch_size=1000)    
    
exp_cfl['exp_test_from_loaded'] = explanations_obj.cf['X']
exp_class = explanations_obj.cf['class'].reshape(-1,)   # equal to model.predict(exp_cfl['exp_test'])
    
# check how many of the target labels we actually achieved
np.sum(target_labels == exp_class)/len(target_labels)  # on the test set we get a success rate of 95.9%
  
# Generate counterfactuals for the train set instances
y_pred = model.predict(X_train)
target_labels = (y_pred == 0)*1   # flip the predicted test set classes so that the explanations explain how to get to the other class
explanations_obj = explainer_new.explain(X_train, Y_t=target_labels, batch_size=1000)        
exp_cfl['exp_train_from_loaded'] = explanations_obj.cf['X']
exp_class = explanations_obj.cf['class'].reshape(-1,)   # equal to model.predict(exp_cfl['exp_test'])
# check how many of the target labels we actually achieved
np.sum(target_labels == exp_class)/len(target_labels)  # on the test set we get a success rate of 98.9%
       
    
"""
    
    
# EoF    
    
    
    
    
    
    
    