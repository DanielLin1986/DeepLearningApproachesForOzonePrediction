# -*- coding: utf-8 -*-
"""
Daniel_Lin

2024-12-01

"""

import math
import pandas as pd
import numpy as np
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import time
import pickle
import datetime
from plot_keras_history import show_history, plot_history
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Switch on XLA to speedup and optimize the machine learning compiler for linear algebra computation.
import tensorflow as tf
tf.config.optimizer.set_jit(True)

from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, Reshape, Embedding, Flatten, Bidirectional, CuDNNGRU, GRU, CuDNNLSTM, LSTM, GlobalMaxPooling1D, BatchNormalization, MaxPooling1D
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, CSVLogger

sampleDataset = pd.read_csv(r'\PathToTheSampleDataset\2020-Northwest_China_Ozone_data.csv', index_col=0)

pm_data = sampleDataset[['O3', 'year',  'doy', 'dem1', 'dem2', 'dem3', 'dem4', 'dem5',
       'dem6','dem7', 'dem8', 'dem9', 'dem91', 'lu1', 'lu2', 'lu3', 'lu4', 'lu5',
       'lu6', 'lu7', 'lu8', 'lu9', 'lu91', 'aod51', 'aod52', 'aod53', 'aod54',
       'aod55', 'aod56','aod57', 'aod58', 'aod59', 'aod591', 'pop1', 'pop2', 'pop3', 'pop4',
       'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop91', 'ph1', 'ph2', 'ph3',
       'ph4', 'ph5', 'ph6','ph7', 'ph8', 'ph9', 'ph91', 'so1', 'so2', 'so3', 'so4', 'so5', 'so6',
       'so7', 'so8', 'so9', 'so91', 'o31', 'o32', 'o33', 'o34', 'o35', 'o36',
       'o37', 'o38', 'o39', 'o391', 'aod471', 'aod472', 'aod473', 'aod474',
       'aod475', 'aod476','aod477', 'aod478', 'aod479', 'aod4791', 'hm', 'pr', 'tem', 'ws',
       'ndvi1', 'ndvi2', 'ndvi3', 'ndvi4', 'ndvi5', 'ndvi6', 'ndvi7', 'ndvi8',
       'ndvi9', 'workd', 'osm1','osm2','osm3','osm4','osm5','osm6','osm7','osm8',
       'osm9','osm91']]

"""
Obtain 4.45 million data samples. Remove negative numbers and perform normalization
"""
pm_data = pm_data.drop(pm_data[(pm_data['O3']>300)].index, axis=0) # Delete the abnormal value contained in the label which is ('O3'>300)

# Partition the data
# We use the data from 2020 as the test set. 
# The data from 2016-2019 as the training and validation sets.
train_set = pm_data.drop(pm_data[(pm_data['year']>=2020)].index, axis=0)

test_set = pm_data.drop(pm_data[(pm_data['year']<2020)].index, axis=0)

# Normalization
scaler = MinMaxScaler()

normalized_train_set = scaler.fit_transform(train_set.values)
normalized_test_set = scaler.transform(test_set.values)

train_x = normalized_train_set[:, 1:]
train_y = normalized_train_set[:, 0]

# The input (X) is reshaped into the 3D format expected by LSTM, i.e. [batch_size, time step, feature]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.20, random_state=39)

test_x = normalized_test_set[:, 1:]
test_y = normalized_test_set[:, 0] 

test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print ("The shape of training set: ")
print (train_x.shape)
print (train_y.shape)
print ("The number of training set:" + str(len(train_x)))
print ("================================")
print ("The shape of validation set: ")
print (valid_x.shape)
print (valid_y.shape)
print ("The number of validation set:" + str(len(valid_x)))
print ("================================")
print ("The shape of test set: ")
print (test_x.shape)
print (test_y.shape)
print ("The number of test set:" + str(len(test_x)))

LOSS_FUNCTION = "mse"
OPTIMIZER = "adam"
BATCH_SIZE = 32
model_name = '2-layer_LSTM'

def build_1LSTM(dnn_size, LOSS_FUNCTION, OPTIMIZER, ACT_FUN, use_dropout):
    inputs = Input(shape=(1,106))
    batch_nor = BatchNormalization()(inputs)
    lstm_0 = CuDNNLSTM(64, return_sequences=True)(batch_nor)
    lstm_1 = CuDNNLSTM(64, return_sequences=True)(lstm_0)
    flatten_0 = Flatten()(lstm_1)
    dense_0 = Dense(64, activation=ACT_FUN)(flatten_0)
    if use_dropout:
        dropout_layer_2 = Dropout(0.2)(dense_0)
        dense_1 = Dense(32, activation=ACT_FUN)(dropout_layer_2)
    else:
        dense_1 = Dense(32, activation=ACT_FUN)(dense_0)

    dense_2 = Dense(1, activation='sigmoid')(dense_1)

    model = Model(inputs=inputs, outputs=dense_2, name=model_name)

    model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=['mae'])
    return model


Neural_model = build_1LSTM(106, LOSS_FUNCTION, OPTIMIZER, 'relu', True)
Neural_model.summary()

"""
Total params: 338,009
Trainable params: 337,133
Non-trainable params: 876
"""

callbacks_list_new = [
    ModelCheckpoint(filepath='./result' + os.sep + Neural_model.name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
                             '_{epoch:02d}_{val_mae:.3f}_{val_loss:.3f}' + '.h5',
                    monitor='val_loss',
                    verbose=1,
                    save_best_only= True,
                    period=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, mode='auto', min_delta=0.00005, min_lr=0, verbose=1),
    EarlyStopping(monitor= 'val_loss',
                  patience=25,
                  verbose=1,
                  mode="auto"),
    TensorBoard(log_dir='logs/',
                batch_size=BATCH_SIZE,
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None),
    CSVLogger('logs/' + os.sep + 'test_model' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')]

time_dnn_a = time.time()
train_history = Neural_model.fit(train_x, train_y,
                                epochs=100,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                validation_data=(valid_x, valid_y),
                                callbacks=callbacks_list_new,
                                verbose=1)

#plot_history(train_history, Neural_model.name, 80)
time_dnn_b = time.time()

probs = Neural_model.predict(test_x, batch_size = BATCH_SIZE, verbose = 1)
evaluated_results = Neural_model.evaluate(test_x, test_y, batch_size= BATCH_SIZE, verbose= 1)
    
print("Normalized results:")
print("-------------------------------------------------")
print(evaluated_results)

print("The MAE:", mean_absolute_error(test_y, probs))
print("The MSE: ", mean_squared_error(test_y, probs))
print("The RMSE:", np.sqrt(mean_squared_error(test_y, probs)))

print("The r2 score is: " + str(r2_score(test_y, probs)))

predicted_data_normalized = normalized_test_set 
predicted_data_normalized[:,0] = probs.flatten() 

predicted_data = scaler.inverse_transform(predicted_data_normalized)
predicted_probs = predicted_data[:,0]
print("-------------------------------------------------")
print("Real results:")
print("-------------------------------------------------")

test_y_original = np.asarray(test_set.iloc[:, 0])

print("The MAE:", mean_absolute_error(test_y_original, predicted_probs))
print("The MSE: ", mean_squared_error(test_y_original, predicted_probs))
print("The RMSE:", np.sqrt(mean_squared_error(test_y_original, predicted_probs)))

print("The r2 score is: " + str(r2_score(test_y_original, predicted_probs)))

print("The time used is: " + str(time_dnn_b - time_dnn_a))