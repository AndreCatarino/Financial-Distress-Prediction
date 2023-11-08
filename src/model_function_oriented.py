from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# BETA

class MLModels:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestRegressor(),
            "SVR": SVR(),
            "LinearRegression": LinearRegression()
        }

    def train(self, model_name, X, y):
        model = self.models.get(model_name)
        if model is not None:
            model.fit(X, y)
        else:
            raise ValueError("Invalid model name.")

    def predict(self, model_name, X):
        model = self.models.get(model_name)
        if model is not None:
            return model.predict(X)
        else:
            raise ValueError("Invalid model name.")

class DeepLearning:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.lstm_model = None

    def build_lstm_model(self, input_shape):
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, input_shape=input_shape))
        self.lstm_model.add(Dense(1, activation='sigmoid'))
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_lstm(self, X_train, y_train):
        if self.lstm_model is not None:
            self.lstm_model.fit(X_train, y_train, epochs=10, verbose=2)
        else:
            raise ValueError("LSTM model has not been built.")

    def predict_lstm(self, X_test):
        if self.lstm_model is not None:
            # Make predictions using the LSTM model
            predictions = self.lstm_model.predict(X_test)
            return np.array([1 if prediction > 0.5 else 0 for prediction in predictions])
            
        else:
            raise ValueError("LSTM model has not been built.")
