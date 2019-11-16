from config.InputDefinition import compileInputList
from config.SampleClasses import *

import sys
import numpy as np
import pandas as pd
import pickle
import json

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from CustomCallback import AdditionalValidationSets

from sklearn.preprocessing import LabelEncoder




### global variables

inputVariableNames = (compileInputList())[:,0]
print("Using these input variables:", inputVariableNames)

usedClasses = []
for key in dict_Classes.keys():
    if dict_Classes[key]["Use"] == True:
        usedClasses.append(key)

print("Using these physical processes as classes of multi-class DNN:", usedClasses)

outputDir = './output/'

### end of global stuff




def load_Numpy(processName, inputSuffix='_norm', workdirName='workdir'):

    """Loads numpy file from work directory. Needs process name (TTbar, QCD, etc.) and file name suffix (default = '_norm')."""

    fileName = './'+workdirName+'/'+processName+inputSuffix+'.npy'
    return np.load(fileName)


def split_TrainTestValidation(processName, percentTrain, percentTest, percentValidation, inputSuffix='_norm', workdirName='workdir'):

    """Splits a given numpy sample into training, test, and validation numpy files. Returns list of file names of train, test, and validation numpy files."""

    if percentTrain+percentTest+percentValidation > 1: sys.exit("Sum of percentages for training, test, and validation samples is greater than 1. Exit.")

    print("Load numpy file for process:", processName)

    loaded_numpy = load_Numpy(processName, inputSuffix, workdirName)
    cardinality = len(loaded_numpy)

    print("Cardinality:", cardinality)

    absoluteTrain = int(cardinality*percentTrain)
    absoluteTest = int(cardinality*percentTest)
    absoluteValidation = int(cardinality*percentValidation)
    
    print("Will split set into train/test/valdiation samples of sizes:", absoluteTrain, absoluteTest, absoluteValidation)

    numpyTrain = loaded_numpy[0:absoluteTrain]
    numpyTest = loaded_numpy[absoluteTrain:absoluteTrain+absoluteTest]
    numpyValidation = loaded_numpy[absoluteTrain+absoluteTest:absoluteTrain+absoluteTest+absoluteValidation]
    
    fileNameTrain = './'+workdirName+'/'+processName+inputSuffix+'_train.npy'
    fileNameTest = './'+workdirName+'/'+processName+inputSuffix+'_test.npy'
    fileNameValidation = './'+workdirName+'/'+processName+inputSuffix+'_validation.npy'

    fileNames = [fileNameTrain, fileNameTest, fileNameValidation]

    print("Saving numpy files...")

    np.save(fileNameTrain, numpyTrain)
    np.save(fileNameTest, numpyTest)
    np.save(fileNameValidation, numpyValidation)

    print("Done saving.")

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


def define_NetworkArchitecture(parameters):

    """Define the NN architecture."""

    layers = parameters['layers']

    my_kernel_regularizer = None
    if parameters['regularizer'] == 'l1':
        my_kernel_regularizer = regularizers.l1(parameters['regularizer_rate'])
    elif parameters['regularizer'] == 'l2':
        my_kernel_regularizer = regularizers.l2(parameters['regularizer_rate'])

    model = Sequential()
    model.add(Dense(layers[0], input_dim=len(inputVariableNames), kernel_regularizer=my_kernel_regularizer, activation='relu'))
    #model.add(BatchNormalization())
    if parameters['dropout']: model.add(Dropout(parameters['dropout_rate']))
    for i in range(len(layers)):
        if i == 0: continue
        model.add(Dense(layers[i], activation='relu', kernel_regularizer=my_kernel_regularizer))
        #model.add(BatchNormalization())
        if parameters['dropout']: model.add(Dropout(parameters['dropout_rate']))
    model.add(Dense(len(parameters['usedClasses']), activation='softmax'))

    my_optimizer = optimizers.Adam(lr=parameters['learning_rate'])
    my_metrics = [metrics.categorical_accuracy]

    model.compile(loss='categorical_crossentropy', optimizer=my_optimizer, metrics=my_metrics)

    # save network architecture to disk before training begins!
    architecture = model.to_json()
    with open(outputDir+'model_arch.json', 'w') as f:
        f.write(architecture)

    print("Neural network architecture SUMMARY:\n", model.summary())

    return model


def train_NN(parameters):

    """Do the actual training of your NN."""

    # split all datasets into training, test, and validation samples!
    for u_cl in parameters['usedClasses']:
        split_TrainTestValidation(u_cl, 0.6, 0.2, 0.2)
        split_TrainTestValidation(u_cl, 0.6, 0.2, 0.2, '_weights')

    # get data for Keras usage!
    data_train = make_DatasetUsableWithKeras(parameters['usedClasses'], 'train')
    data_test = make_DatasetUsableWithKeras(parameters['usedClasses'], 'test')
    data_validation = make_DatasetUsableWithKeras(parameters['usedClasses'], 'validation')

    # initialize your own custom history callback in which training set and validation set are evaluated after each epoch in the same way!
    customHistory = AdditionalValidationSets([
        (data_train['values'], data_train['encodedLabels'], data_train['weights'], 'train'),
        (data_validation['values'], data_validation['encodedLabels'], data_validation['weights'], 'valid')
    ])

    # initialize checkpointer callback
    filePathCheckPoints = outputDir+'checkpoints/checkpoint-{epoch:03d}.h5'
    checkpointer = ModelCheckpoint(filePathCheckPoints, verbose=1, save_weights_only=True, period=10) # lwtnn only supports json+hdf5 format, not hdf5 standalone

    # train!
    model = define_NetworkArchitecture(parameters)
    history = model.fit(data_train['values'], data_train['encodedLabels'], sample_weight=data_train['weights'], epochs=parameters['epochs'], batch_size=parameters['batch_size'], shuffle=True, validation_data=(data_validation['values'], data_validation['encodedLabels'], data_validation['weights']), callbacks=[customHistory, checkpointer])
    print("Model history:\n", history.history)

    # save final model to disk!
    model.save(outputDir+'model.h5')
    model.save_weights(outputDir+'model_weights.h5')
    with open(outputDir+'model_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    with open(outputDir+'model_customHistory.pkl', 'wb') as f:
        pickle.dump(customHistory.history, f)
    print("Saved model to disk.")


def dump_ParametersIntoJsonFile(parameters):

    """Saves NN parameters to json file."""

    param_json = json.dumps(parameters)
    with open(outputDir+'parameters.json', 'w') as f:
        f.write(param_json)


if __name__ == '__main__':

    parameters = {
        'usedClasses': ['tW_signal', 'tW_other', 'tChannel', 'sChannel', 'TTbar', 'WJets', 'DYJets', 'Diboson', 'QCD'],
        'layers': [64, 64],
        'dropout': True,
        'dropout_rate': 0.6,
        'epochs': 800,
        'batch_size': 65536,
        'learning_rate': 0.0001,
        'regularizer': '', # either 'l1' or 'l2' or just ''
        'regularizer_rate': 0.01
    }

    dump_ParametersIntoJsonFile(parameters)
    train_NN(parameters)
    print("Done.")
