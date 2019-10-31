from config.InputDefinition import compileInputList
from config.SampleClasses import *

import sys
import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
from keras import metrics
from CustomCallback import AdditionalValidationSets

from sklearn.preprocessing import LabelEncoder




### global variables

inputVariableNames = (compileInputList())[:,0]
print "Using these input variables:",inputVariableNames

usedClasses = []
for key in dict_Classes.keys():
    if dict_Classes[key]["Use"] == True:
        usedClasses.append(key)

print "Using these physical processes as classes of multi-class DNN:",usedClasses

### end of global stuff




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

        fileNameWeights = './'+workdirName+'/'+u_cl+'_weights_'+sample_type+'.npy'
        weightsArray = np.load(fileNameWeights)
        dataFrame['EventWeight'] = weightsArray

        dataFrame['Class'] = u_cl

        listOfDataFrames.append(dataFrame)

    completeDataFrame = pd.concat(listOfDataFrames, ignore_index=True, sort=False)

    return completeDataFrame


def make_DatasetUsableWithKeras(used_classes, sample_type, inputSuffix='_norm', workdirName='workdir'): # sample_type = 'train', 'test', or 'validation'

    """Returns a dictionary containing data values, string labels, and encoded labels which can directly be used with the model.fit() function of Keras."""

    # In the following, do everything as described here:
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    # load dataset
    data = prepare_Dataset(used_classes, sample_type, inputSuffix, workdirName).values
    data_values = data[:,0:-2] # input vectors for NN
    data_labels = data[:,-1] # classes associated to each event, given in string format
    data_weights = data[:,-2] # event weights

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(data_labels)
    encoded_data_labels = encoder.transform(data_labels)

    # convert integers to dummy values (i.e. one hot encoded)
    data_encodedLabels = np_utils.to_categorical(encoded_data_labels)

    result = {"values": data_values, "labels": data_labels, "encodedLabels": data_encodedLabels, "weights": data_weights}

    return result


def define_NetworkArchitecture(used_classes):

    """Define the NN architecture."""

    layers = [256,256]

    model = Sequential()
    model.add(Dense(layers[0], input_dim=len(inputVariableNames), activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.6))
    for i in range(len(layers)):
        if i == 0: continue
        model.add(Dense(layers[i], activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.6))
    model.add(Dense(len(used_classes), activation='softmax'))

    my_optimizer = optimizers.Adam(lr=0.001)
    my_metrics = [metrics.categorical_accuracy]

    model.compile(loss='categorical_crossentropy', optimizer=my_optimizer, metrics=my_metrics)

    print "Neural network architecture SUMMARY:"
    print model.summary()

    return model


def train_NN(parameters):

    """Do the actual training of your NN."""

    # split all datasets into training, test, and validation samples!
    for u_cl in usedClasses:
        split_TrainTestValidation(u_cl, 0.6, 0.2, 0.2)
        split_TrainTestValidation(u_cl, 0.6, 0.2, 0.2, '_weights')

    # get data for Keras usage!
    data_train = make_DatasetUsableWithKeras(usedClasses, 'train')
    data_test = make_DatasetUsableWithKeras(usedClasses, 'test')
    data_validation = make_DatasetUsableWithKeras(usedClasses, 'validation')

    # initialize your own custom history callback in which training set and validation set are evaluated after each epoch in the same way!
    customHistory = AdditionalValidationSets([
        (data_train['values'], data_train['encodedLabels'], data_train['weights'], 'train'),
        (data_validation['values'], data_validation['encodedLabels'], data_validation['weights'], 'valid')
    ])

    # train!
    model = define_NetworkArchitecture(usedClasses)
    history = model.fit(data_train['values'], data_train['encodedLabels'], sample_weight=data_train['weights'], epochs=5, batch_size=65536, shuffle=True, validation_data=(data_validation['values'], data_validation['encodedLabels'], data_validation['weights']), callbacks=[customHistory])
    print history


if __name__ == '__main__':

    train_NN(None)
    print "Done."
