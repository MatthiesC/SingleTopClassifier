from config.InputDefinition import compileInputList
from config.SampleClasses import *

import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense


### global variables

inputVariableNames = (compileInputList())[:,0]
#print inputVariableNames


def load_Numpy(processName, inputSuffix='_norm', workdirName='workdir'):

    """Loads numpy file from work directory. Needs process name (TTbar, QCD, etc.) and file name suffix (default = '_norm')."""

    fileName = './'+workdirName+'/'+processName+inputSuffix+'.npy'
    return np.load(fileName)


def split_TrainTestValidation(processName, percentTrain, percentTest, percentValidation, inputSuffix='_norm', workdirName='workdir'):

    """Splits a given numpy sample into training, test, and validation numpy files. Returns list of file names of train, test, and validation numpy files."""

    if percentTrain+percentTest+percentValidation > 1: sys.exit("Sum of percentages for training, test, and validation samples is greater than 1. Exit.")

    print "Load numpy file for process:",processName

    loaded_numpy = load_Numpy(processName, inputSuffix, workdirName)
    cardinality = len(loaded_numpy)

    print "Cardinality:",cardinality

    absoluteTrain = int(cardinality*percentTrain)
    absoluteTest = int(cardinality*percentTest)
    absoluteValidation = int(cardinality*percentValidation)
    
    print "Will split set into train/test/valdiation samples of sizes:",absoluteTrain,absoluteTest,absoluteValidation

    numpyTrain = loaded_numpy[0:absoluteTrain]
    numpyTest = loaded_numpy[absoluteTrain:absoluteTrain+absoluteTest]
    numpyValidation = loaded_numpy[absoluteTrain+absoluteTest:absoluteTrain+absoluteTest+absoluteValidation]
    
    fileNameTrain = './'+workdirName+'/'+processName+inputSuffix+'_train.npy'
    fileNameTest = './'+workdirName+'/'+processName+inputSuffix+'_test.npy'
    fileNameValidation = './'+workdirName+'/'+processName+inputSuffix+'_validation.npy'

    fileNames = [fileNameTrain, fileNameTest, fileNameValidation]

    print "Saving numpy files..."

    np.save(fileNameTrain, numpyTrain)
    np.save(fileNameTest, numpyTest)
    np.save(fileNameValidation, numpyValidation)

    print "Done saving."

    return fileNames


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

    split_TrainTestValidation('QCD', 0.6, 0.2, 0.2)
