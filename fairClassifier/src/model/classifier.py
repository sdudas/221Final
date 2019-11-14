from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Conv1D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'configs' ))
import constant

"""
Classifier class contains method to create NN classifier
"""
class Classifier:

    def __init__(self):
        self.random_forest_params = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 50, 
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6,
              'max_leaf_nodes': None}

        self.type = 'feed forward'
        pass

    """
    Create a NN classifier given the shape number of features as input tensor.
    Returns the created model
    """
    def create_nn_classifier(self,n_features):
        if self.type == 'tree':
            return RandomForestClassifier(**self.random_forest_params)

        inputs = Input(shape=(n_features,))

        dense1 = Dense(constant.UNIT, activation='relu')(inputs)
        dropout1 = Dropout(constant.DROP_RATE)(dense1)
        # dense2 = Dense(constant.UNIT, activation='relu')(dropout1)
        # dropout2 = Dropout(constant.DROP_RATE)(dense2)
        # dense3 = Dense(constant.UNIT, activation="relu")(dropout2)
        # dropout3 = Dropout(constant.DROP_RATE)(dense3)
        outputs = Dense(constant.OUTPUT_UNIT, activation='sigmoid')(dropout1)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
