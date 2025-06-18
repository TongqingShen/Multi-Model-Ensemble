# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
from skopt import BayesSearchCV  
from sklearn.neural_network import MLPRegressor  
from sklearn.metrics import mean_squared_error  
from skopt.space import Real, Integer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import RandomForestRegressor  

# import data
# base period
obs = pd.read_excel(io=r'F:\Research\SCI\res\his\base_obs_pr.xlsx',header=None)
train = pd.read_excel(io=r'F:\Research\SCI\res\his\train_mods_corr_pr.xlsx',header=0)
valid = pd.read_excel(io=r'F:\Research\SCI\res\his\valid_mods_corr_pr.xlsx',header=0)
data_obs = np.array(obs)
data_train = np.array(train)
data_valid = np.array(valid)
y_train = data_obs[:372]
# future data
ssp126 = pd.read_excel(io=r'F:\Research\SCI\res\future\ssp_126_corr_pr.xlsx',header=0)
ssp245 = pd.read_excel(io=r'F:\Research\SCI\res\future\ssp_245_corr_pr.xlsx',header=0) 
ssp585 = pd.read_excel(io=r'F:\Research\SCI\res\future\ssp_585_corr_pr.xlsx',header=0)

mods126 = np.array(ssp126)
mods245 = np.array(ssp245)
mods585 = np.array(ssp585)
# Data normalization function
def normalize_data(data,data_min,data_max):  
    normalized_data = (data - data_min) / (data_max - data_min)  
    return normalized_data
# Data inverse normalization function
def denormalize_data(normalized_data, data_min, data_max):  
    denormalized_data = normalized_data * (data_max - data_min) + data_min  
    return denormalized_data  
  
X_train_scaled = normalize_data(data_train,np.min(y_train),np.max(y_train))
y_train_scaled = normalize_data(y_train,np.min(y_train),np.max(y_train))
X_valid_scaled = normalize_data(data_valid,np.min(y_train),np.max(y_train))

mods126_scaled = normalize_data(mods126,np.min(y_train),np.max(y_train))
mods245_scaled = normalize_data(mods245,np.min(y_train),np.max(y_train))
mods585_scaled = normalize_data(mods585,np.min(y_train),np.max(y_train))
#%% Weighted average method
rmse = np.ones((6,1))
for i in range(6):
    rmse[i,0] = np.sqrt(mean_squared_error(y_train,data_train[:,i]))
wei = (1/rmse)/np.sum(1/rmse)#Weight
base_wei = np.dot(np.append(data_train,data_valid,axis=0), wei)#matrix multiplication
# future prediction
ssp126_wei = np.dot(mods126, wei)
ssp245_wei = np.dot(mods245, wei)
ssp585_wei = np.dot(mods585, wei)
#%% BP
# Define BP regression model
model_BP = MLPRegressor()

# Define parameter space
param_BP = {
    'hidden_layer_sizes': Integer(low=20, high=100, name='hidden_layer_sizes'),
    'activation': ['relu', 'tanh', 'logistic'],  
    'solver': ['lbfgs', 'adam'],  
    'alpha': Real(low=1e-4, high=1e-1, prior='log-uniform', name='alpha'), 
    'learning_rate': ['constant', 'invscaling', 'adaptive'], 
    'early_stopping': [True],  
    'max_iter': [500]  
}

# Create BayesSearchCV object
search_BP = BayesSearchCV(model_BP, param_BP, cv=5, n_iter=50, n_jobs=-1)

# Search and fit on the training set
search_BP.fit(X_train_scaled, y_train_scaled)

# Output the optimal parameter combination
print("Best parameters:", search_BP.best_params_)

BP_fit = search_BP.predict(np.append(X_train_scaled,X_valid_scaled,axis=0))
base_BP = denormalize_data(BP_fit, np.min(y_train),np.max(y_train))
# future prediction
ssp126_fit_BP  = search_BP.predict(mods126_scaled)
ssp245_fit_BP  = search_BP.predict(mods245_scaled)
ssp585_fit_BP  = search_BP.predict(mods585_scaled)
ssp126_BP = denormalize_data(ssp126_fit_BP, np.min(y_train),np.max(y_train))
ssp245_BP = denormalize_data(ssp245_fit_BP, np.min(y_train),np.max(y_train))
ssp585_BP = denormalize_data(ssp585_fit_BP, np.min(y_train),np.max(y_train))
#%% LSTM
# Create LSTM model
X_train_reshaped = np.reshape(X_train_scaled , (372, 1, 6))
X_valid_reshaped = np.reshape(X_valid_scaled , (60, 1, 6))

mods126_reshaped = np.reshape(mods126_scaled , (1032, 1, 6))
mods245_reshaped = np.reshape(mods245_scaled , (1032, 1, 6))
mods585_reshaped = np.reshape(mods585_scaled , (1032, 1, 6))
# Build model function
def create_lstm_model(hidden_units=128, epochs=10, batch_size=32):  
    model = Sequential()  
    model.add(LSTM(hidden_units, input_shape=(1,6)))  
    model.add(Dense(1))  
    model.compile(loss='mean_squared_error', optimizer='adam')  
    return model  
  
# Define parameter space 
param_LSTM = {   
    'batch_size': Integer(32,64),  
    'epochs': Integer(10,100)
}  
  
# Packaging LSTM model with KerasRegressor
keras_regressor = KerasRegressor(build_fn=create_lstm_model, verbose=0)  
  
# Using BayesSearchCV for hyperparameter search 
search_LSTM = BayesSearchCV(  
    keras_regressor,  
    param_LSTM,  
    n_iter=10, 
    cv=5,   
    scoring='neg_mean_squared_error',  
    verbose=2   
)  
  
# Train the model and search for the best hyperparameters  
search_LSTM.fit(X_train_reshaped, y_train_scaled)  
  
# Output the optimal parameter combination 
print("Best parameters found:", search_LSTM.best_params_)  

LSTM_fit = search_LSTM.predict(np.concatenate((X_train_reshaped,X_valid_reshaped),axis=0))  
base_LSTM = denormalize_data(LSTM_fit, np.min(y_train),np.max(y_train)) #denormalize
# future prediction
ssp126_fit_LSTM  = search_LSTM.predict(mods126_reshaped)
ssp245_fit_LSTM  = search_LSTM.predict(mods245_reshaped)
ssp585_fit_LSTM  = search_LSTM.predict(mods585_reshaped)
ssp126_LSTM = denormalize_data(ssp126_fit_LSTM, np.min(y_train),np.max(y_train))
ssp245_LSTM = denormalize_data(ssp245_fit_LSTM, np.min(y_train),np.max(y_train))
ssp585_LSTM = denormalize_data(ssp585_fit_LSTM, np.min(y_train),np.max(y_train))
#%% RF
# Define Random Forest Regression Model
rf_model = RandomForestRegressor()  
  
# Define parameter space  
params_RF = {  
    'n_estimators': Integer(50, 100),   
    'max_depth': Integer(3, 25), 
    'min_samples_split': Integer(2, 10),  
    'min_samples_leaf': Integer(1, 4),    
    'max_features': ('sqrt', 'log2', None),  
}  
  
# Using BayesSearchCV for hyperparameter search  
search_RF = BayesSearchCV(  
    estimator=rf_model,  
    search_spaces=params_RF,  
    n_iter=20, 
    cv=5,    
    scoring='neg_mean_squared_error',   
    verbose=2  
)  
  
# Fit data and search for optimal parameters  
search_RF.fit(X_train_scaled, y_train_scaled)  
print("Best parameters found:", search_RF.best_params_)  

RF_fit = search_RF.predict(np.append(X_train_scaled,X_valid_scaled,axis=0))  
base_RF = denormalize_data(RF_fit, np.min(y_train),np.max(y_train))
# future prediction
ssp126_fit_RF  = search_RF.predict(mods126_scaled)
ssp245_fit_RF  = search_RF.predict(mods245_scaled)
ssp585_fit_RF  = search_RF.predict(mods585_scaled)
ssp126_RF = denormalize_data(ssp126_fit_RF, np.min(y_train),np.max(y_train))
ssp245_RF = denormalize_data(ssp245_fit_RF, np.min(y_train),np.max(y_train))
ssp585_RF = denormalize_data(ssp585_fit_RF, np.min(y_train),np.max(y_train))