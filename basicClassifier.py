from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'configs' ))
import constant

"""
Classifier class contains method to create NN classifier
"""
class Classifier:

    def __init__(self):
        pass

    """
    Create a NN classifier given the shape number of features as input tensor.
    Returns the created model
    """
    def create_nn_classifier(self,num_features):
        inputs = Input(shape=(num_features,))
        dense1 = Dense(constant.UNIT, activation='relu')(inputs)
        dropout1 = Dropout(constant.DROP_RATE)(dense1)
        dense2 = Dense(constant.UNIT, activation='relu')(dropout1)
        dropout2 = Dropout(constant.DROP_RATE)(dense2)
        dense3 = Dense(constant.UNIT, activation="relu")(dropout2)
        dropout3 = Dropout(constant.DROP_RATE)(dense3)
        outputs = Dense(constant.OUTPUT_UNIT, activation='sigmoid')(dropout3)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model


    """
    Create a NN classifier given the data which is a dicitonary of x_train, y_train, x_test, y_test
    and the shape number of features as input tensor.
    Returns the score based on the test data.
    """
    def create_simple_classifier(self, data, num_features):
        model = Sequential()
        model.add(Dense(constant.UNIT, input_dim=num_features, activation='relu'))
        model.add(Dropout(constant.DROP_RATE))
        for i in range(constant.NUM_LAYERS - 1):
            model.add(Dense(constant.UNIT, activation='relu'))
            model.add(Dropout(constant.DROP_RATE))

        model.add(Dense(constant.OUTPUT_UNIT, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        x_trainD = data['x_train']
        y_trainD = data['y_train']
        x_testD = data['x_test']
        y_testD = data['y_test']
        model.fit(x_trainD, y_trainD,
                epochs=constant.NUM_EPOCH,
                batch_size=constant.UNIT)
        score = model.evaluate(x_testD, y_testD, batch_size=constant.UNIT)
        
