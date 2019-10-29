from config.InputDefinition import compileInputList
from config.SampleClasses import *

import sys
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense




### global variables

inputVariableNames = (compileInputList())[:,0]
print "Using these input variables:",inputVariableNames

usedClasses = []
for key in dict_Classes.keys():
    if dict_Classes[key]["Use"] == True:
        usedClasses.append(key)
print "Using these physical processes as classes of multi-class DNN:",usedClasses




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


def prepare_Dataset(used_classes, sample_type, inputSuffix='_norm', workdirName='workdir'): # sample_type = 'train', 'test', or 'validation'

    """Returns a gigantic pandas dataframe, containing all events to be trained/tested/validated on."""

    listOfDataFrames = []

    for u_cl in used_classes:
        fileName = './'+workdirName+'/'+u_cl+inputSuffix+'_'+sample_type+'.npy'
        dataArray = np.load(fileName)
        dataFrame = pd.DataFrame(data=dataArray, columns=inputVariableNames)
        dataFrame['Class'] = u_cl
        listOfDataFrames.append(dataFrame)

    completeDataFrame = pd.concat(listOfDataFrames, ignore_index=True, sort=False)

    return completeDataFrame


def define_Model():

    """Define the NN architecture."""

    model = Sequential()
    model.add(Dense(512, input_dim=len(inputVariableNames), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(usedClasses), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_NN():

    """Do the actual training of your NN."""

    model = define_Model()
    #model.fit(X, y, epochs=10, batch_size=1024)


if __name__ == '__main__':

    for c in usedClasses:
        split_TrainTestValidation(c, 0.6, 0.2, 0.2)

    print prepare_Dataset(usedClasses, 'train')

    print "Done."
