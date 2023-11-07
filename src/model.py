import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def fit_predict(self, X, y):
        pass

class NaiveBayesClassifier(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

    def fit(self):
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def fit_predict(self):
        return self.fit().predict()

class XGBoostModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
    
    def fit(self):
        self.model = xgb.XGBClassifier()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def fit_predict(self):
        return self.fit().predict()

class AdaBoostModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

    def fit(self):
        self.model = AdaBoostClassifier()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def fit_predict(self):
        return self.fit().predict()

class ANN_classifier(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

    def fit(self):
        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_dim=self.X_train.shape[-1]))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=2)

    def predict(self):
        return self.model.predict(self.X_test)

    def fit_predict(self):
        return self.fit().predict()
        
class LSTMModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test

    def fit(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(LSTM(50), return_sequences=True)
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=2, shuffle=False)

    def predict(self):
        return self.model.predict(self.X_test)

    def fit_predict(self):
        return self.fit().predict()
