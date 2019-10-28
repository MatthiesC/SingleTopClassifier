from config.InputDefinition import compileInputList

import numpy as np

from keras.models import Sequential
from keras.layers import Dense


### global variables

inputVariableNames = (compileInputList())[:,0]
#print inputVariableNames


def define_Model():

    """Define the NN architecture."""

    model = Sequential()
    model.add(Dense(512, input_dim=len(inputVariableNames), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_NN():

    """Do the actual training of your NN."""

    model = define_Model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(X, y, epochs=10, batch_size=1024)


if __name__ == '__main__':

    train_NN()
