# -*- coding: utf-8 -*-
"""
Daniel_Lin

2024-12-01

"""

import math
import time
import os
import numpy as np
import pandas as pd
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import datetime
from plot_keras_history import show_history, plot_history
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import layers
from tensorflow import keras
import pickle
# Switch on XLA to speedup and optimize the machine learning compiler for linear algebra computation.
import tensorflow as tf
tf.config.optimizer.set_jit(True)

from keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, Reshape, Embedding, Flatten, Bidirectional, CuDNNGRU, GRU, CuDNNLSTM, LSTM, GlobalMaxPooling1D, BatchNormalization, MaxPooling1D
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, CSVLogger

sampleDataset = pd.read_csv(r'\PathToTheSampleDataset\2020-Northwest_China_Ozone_data.csv', index_col=0)

# Predicting O3
pm_data=sampleDataset[['O3', 'year',  'doy', 'dem1', 'dem2', 'dem3', 'dem4', 'dem5',
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
train_set = pm_data.drop(pm_data[(pm_data['year']>=2020)].index, axis=0)

test_set = pm_data.drop(pm_data[(pm_data['year']<2020)].index, axis=0)

# Normalization
scaler = MinMaxScaler()

normalized_train_set = scaler.fit_transform(train_set.values)
normalized_test_set = scaler.transform(test_set.values)

train_x = normalized_train_set[:, 1:]
train_y = normalized_train_set[:, 0]

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.20, random_state=39)

test_x = normalized_test_set[:, 1:]
test_y = normalized_test_set[:, 0] 

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
valid_x = valid_x.reshape((valid_x.shape[0], valid_x.shape[1], 1))
test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

"""
Our model processes a tensor of shape (batch size, sequence length, features), 
where sequence length is the number of time steps and features is each input timeseries.
"""

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

BATCH_SIZE = 32
LOSS_FUNCTION = "mse"
OPTIMIZER = "adam"
model_name = 'Transformer-2-block'

def SavedPickle(path, file_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(file_to_save, handle)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

input_shape = train_x.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=['mae'])

model.summary()

"""
Total params: 22,547
Trainable params: 22,547
Non-trainable params: 0
"""

callbacks_list_new = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_delta=0.00001, mode='auto', min_lr=0, verbose=1),
    ModelCheckpoint(filepath='./result' + os.sep + model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
                             '_{epoch:02d}_{val_mae:.6f}_{val_loss:.6f}' + '.h5',
                    monitor='val_loss',
                    verbose=1,
                    save_best_only= True,
                    period=1),
    EarlyStopping(monitor= 'val_loss',
                  patience=25,
                  verbose=1,
                  mode="auto"),
    TensorBoard(log_dir='logs/' + model_name,
                batch_size=BATCH_SIZE,
                write_graph=True,
                write_grads=True,
                write_images=True,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None),
    CSVLogger('logs/' + os.sep + model_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')]

time_dnn_a = time.time()
train_history = model.fit(train_x, train_y,
                                 epochs=100,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list_new,
                                 verbose=1)

time_dnn_b = time.time()

probs = model.predict(test_x, batch_size=BATCH_SIZE, verbose=1)
evaluated_results = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE, verbose=1)

print("Normalized results:")
print("-------------------------------------------------")
print(evaluated_results)

print("The MAE:", mean_absolute_error(test_y, probs))
print("The MSE: ", mean_squared_error(test_y, probs))
print("The RMSE:", np.sqrt(mean_squared_error(test_y, probs)))

print("The r2 score is: " + str(r2_score(test_y, probs)))

predicted_data_normalized = normalized_test_set 
predicted_data_normalized[:,0] = probs.flatten

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
