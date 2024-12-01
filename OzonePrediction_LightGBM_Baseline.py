# -*- coding: utf-8 -*-
"""
Daniel_Lin

2024-12-01

"""
import os
import pickle
import math
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
#import joblib
#import seaborn as sns
#import xgboost as xgb

print("Load the data...")

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
pm_data = pm_data.drop(pm_data[(pm_data['O3']>300)].index, axis=0) # Delete the abnormal value contained in the label

# Prediction

train_set = pm_data.drop(pm_data[(pm_data['year']>=2020)].index, axis=0)

test_set = pm_data.drop(pm_data[(pm_data['year']<2020)].index, axis=0)

train_x, train_y = np.asarray(train_set.iloc[:, 1:]), np.asarray(train_set.iloc[:, 0])

test_x, test_y = np.asarray(test_set.iloc[:, 1:]), np.asarray(test_set.iloc[:, 0])

# shuffle the data.
np.random.seed(39)
np.random.shuffle(train_x)
np.random.seed(39)
np.random.shuffle(train_y)

np.random.seed(39)
np.random.shuffle(test_x)
np.random.seed(39)
np.random.shuffle(test_y)


print ("The shape of training set: ")
print (train_x.shape)
print (train_y.shape)
print ("The number of training set:" + str(len(train_x)))
print ("================================")
#print ("The shape of validation set: ")
#print (valid_x.shape)
#print (valid_y.shape)
#print ("The number of validation set:" + str(len(valid_x)))
print ("================================")
print ("The shape of test set: ")
print (test_x.shape)
print (test_y.shape)
print ("The number of test set:" + str(len(test_x)))

def saveModel(model_name, saved_path, model_to_save):
    with open (saved_path + model_name, 'wb') as pkM:
        pickle.dump(model_to_save, pkM)

"""
print("=============================Random Forest============================")
rf_t_a = time.time()
rf_regr = RandomForestRegressor(verbose=1,
                                n_jobs=42, n_estimators=250)
rf_regr.fit(train_x, train_y)
rf_t_b = time.time()
print("Time needed: " + str(rf_t_b-rf_t_a))
y_predict = rf_regr.predict(test_x)

print("The r2 score is: " + str(r2_score(test_y, y_predict)))
# MAE
print("The mean_absolute_error:", mean_absolute_error(test_y, y_predict))
# RMSE
print("The RMSE:", getRMSE(test_y, y_predict))
"""

#saveModel("randomForest_regr.pkl", saved_dir + os.sep, rf_regr)
#saved_dir = "D:\\Light-weight_codeBERT\\Benchmark-2022-07-11\\predictiong_models"

print("=============================Start benchmarking============================")
print("=============================LightGMB============================")
t_lgb_time = time.time()

lgb_regr = lgb.LGBMRegressor()
lgb_regr.get_params()

"""
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': -1,
 'num_leaves': 31,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0}

"""

lgb_regr = lgb.LGBMRegressor(boosting_type='dart', num_leaves=2000, max_depth=-1, learning_rate=0.05, n_estimators=2000, 
                        subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, 
                       min_child_weight=0.001, min_child_samples=100, subsample=1, subsample_freq=0, 
                        colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, 
                        importance_type='split')
lgb_regr.fit(train_x, train_y)
t_lgb_time_after = time.time()
print("Time needed: " + str(t_lgb_time_after-t_lgb_time))
y_predict_lgb = lgb_regr.predict(test_x)
print("Score: " + str(lgb_regr.score(test_x, test_y)))
print("The r2 score is: " + str(r2_score(test_y, y_predict_lgb)))
print("The mean_absolute_error:", mean_absolute_error(test_y, y_predict_lgb))
print("The mean_squared_error:", mean_squared_error(test_y, y_predict_lgb))
# RMSE
print("The RMSE:", np.sqrt(mean_squared_error(test_y, y_predict_lgb)))

#print(lgb_regr.feature_importance())
#saveModel("LightGMB_regr_106_acrossChina.pkl", r"D:" + os.sep, lgb_regr)
