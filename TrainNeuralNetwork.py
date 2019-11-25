from config.InputDefinition import compileInputList
from config.SampleClasses import *

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json

from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from CustomCallback import AdditionalValidationSets
from Plotting import get_Parameters

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from datetime import datetime


### global variables

inputVariableNames = (compileInputList())[:,0]
print("Using these input variables:", inputVariableNames)

# use a time tag to identify trained models later (for details, look up the parameters.json in the output dir)
timeTag = datetime.now()
timeTag = timeTag.strftime('%y-%m-%d-%H-%M-%S')
outputDir = './outputs/dnn_'+timeTag+'/'
checkpointDir = 'checkpoints/' #subfolder of outputDir

### end of global stuff




def main():

    """The main function."""

    print("Creating output directory:", outputDir)
    os.makedirs(outputDir+checkpointDir, exist_ok=True)
    os.makedirs(outputDir+'workdir/', exist_ok=True)

    parameters = {
        #'usedClasses': ['tW_signal', 'tW_other', 'tChannel', 'sChannel', 'TTbar', 'WJets', 'DYJets', 'Diboson', 'QCD'],
        'usedClasses': ['tW_signal', 'tW_bkg_TopToHadAndWToTau', 'tW_bkg_Else', 'tChannel', 'sChannel', 'TTbar', 'WJets', 'DYJets', 'Diboson', 'QCD'],
        #'usedClasses': ['tW_signal', 'tW_other', 'TTbar', 'WJets', 'DYJets'],
        #'usedClasses': ['tW_signal', 'tW_bkg_TopToHadAndWToTau', 'tW_bkg_Else', 'TTbar', 'WJets', 'DYJets'],
        'splits': { 'train': 0.6, 'test': 0.2, 'validation': 0.2 },
        'layers': [16, 16],
        'dropout': True,
        'dropout_rate': 0.5,
        'epochs': 1,
        'batch_size': 65536,
        'learning_rate': 0.001, #Adam default: 0.001
        'regularizer': '', # either 'l1' or 'l2' or just ''
        'regularizer_rate': 0.01
    }

    dump_ParametersIntoJsonFile(parameters)
    train_NN(parameters)
    print("Done.")







def load_Numpy(processName, inputSuffix='_norm'):

    """Loads numpy file from work directory. Needs process name (TTbar, QCD, etc.) and file name suffix (default = '_norm')."""

    fileName = './workdir/'+processName+inputSuffix+'.npy'
    return np.load(fileName)


def split_TrainTestValidation(processName, percentTrain, percentTest, percentValidation, shuffle_seed, inputSuffix='_norm'):

    """Splits a given numpy sample into training, test, and validation numpy files. Returns list of file names of train, test, and validation numpy files."""

    if percentTrain+percentTest+percentValidation > 1: sys.exit("Sum of percentages for training, test, and validation samples is greater than 1. Exit.")

    print("Load numpy file for process:", processName)

    loaded_numpy = load_Numpy(processName, inputSuffix)
    cardinality = len(loaded_numpy)

    print("Cardinality:", cardinality)

    absoluteTrain = int(cardinality*percentTrain)
    absoluteTest = int(cardinality*percentTest)
    absoluteValidation = int(cardinality*percentValidation)
    
    print("Will split set into train/test/valdiation samples of sizes:", absoluteTrain, absoluteTest, absoluteValidation)

    # need to shuffle since hadd presumably does not shuffle... (e.g. WJets is ordered by Pt sample bins, TTbar is ordered by Mtt)
    # shuffle, but with fixed random seed so that inputs and sample weights are not shuffled differently!!! Should do this more elegant and without fixed seed in the future...
    loaded_numpy = shuffle(loaded_numpy, random_state=shuffle_seed)

    numpyTrain = loaded_numpy[0:absoluteTrain]
    numpyTest = loaded_numpy[absoluteTrain:absoluteTrain+absoluteTest]
    numpyValidation = loaded_numpy[absoluteTrain+absoluteTest:absoluteTrain+absoluteTest+absoluteValidation]
    
    fileNameTrain = outputDir+'workdir/'+processName+inputSuffix+'_train.npy'
    fileNameTest = outputDir+'workdir/'+processName+inputSuffix+'_test.npy'
    fileNameValidation = outputDir+'workdir/'+processName+inputSuffix+'_validation.npy'

    fileNames = [fileNameTrain, fileNameTest, fileNameValidation]

    print("Saving numpy files...")

    np.save(fileNameTrain, numpyTrain)
    np.save(fileNameTest, numpyTest)
    np.save(fileNameValidation, numpyValidation)

    print("Done saving.")

    return fileNames


def prepare_Dataset(used_classes, sample_type, inputSuffix='_norm'): # sample_type = 'train', 'test', or 'validation'

    """Returns a gigantic pandas dataframe, containing all events to be trained/tested/validated on."""

    listOfDataFrames = []

    for u_cl in used_classes:

        fileName = outputDir+'workdir/'+u_cl+inputSuffix+'_'+sample_type+'.npy'
        dataArray = np.load(fileName)
        dataFrame = pd.DataFrame(data=dataArray, columns=inputVariableNames)

        fileNameWeights = outputDir+'workdir/'+u_cl+'_weights_'+sample_type+'.npy'
        weightsArray = np.load(fileNameWeights)
        dataFrame['EventWeight'] = weightsArray

        dataFrame['Class'] = u_cl

        listOfDataFrames.append(dataFrame)

    completeDataFrame = pd.concat(listOfDataFrames, ignore_index=True, sort=False)

    return completeDataFrame


def make_DatasetUsableWithKeras(used_classes, sample_type, inputSuffix='_norm'): # sample_type = 'train', 'test', or 'validation'

    """Returns a dictionary containing data values, string labels, and encoded labels which can directly be used with the model.fit() function of Keras."""

    # In the following, do everything as described here:
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    # load dataset
    data = prepare_Dataset(used_classes, sample_type, inputSuffix).values
    data_values = data[:,0:-2] # input vectors for NN
    data_labels = data[:,-1] # classes associated to each event, given in string format
    data_weights = data[:,-2] # event weights

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(data_labels)
    encoded_data_labels = encoder.transform(data_labels)
    print("Label encoder:", encoded_data_labels)

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


def predict_Labels(parameters, model, data_train, data_test, data_validation):

    """Predict labels of train/test/validation datasets using the trained model."""

    predDir = outputDir+'predictions/'
    os.makedirs(predDir, exist_ok=True)

    print("Predicting labels of training dataset...")
    pred_train = model.predict(data_train['values'])
    np.save(predDir+'pred_train.npy', pred_train)
    for u_cl in parameters['usedClasses']:
        tmp = pred_train[data_train['labels'] == u_cl]
        np.save(predDir+'pred_train__'+str(u_cl)+'.npy', tmp)

    print("Predicting labels of test dataset...")
    pred_test = model.predict(data_test['values'])
    np.save(predDir+'pred_test.npy', pred_test)
    for u_cl in parameters['usedClasses']:
        tmp = pred_test[data_test['labels'] == u_cl]
        np.save(predDir+'pred_test__'+str(u_cl)+'.npy', tmp)

    print("Predicting labels of validation dataset...")
    pred_validation = model.predict(data_validation['values'])
    np.save(predDir+'pred_validation.npy', pred_validation)
    for u_cl in parameters['usedClasses']:
        tmp = pred_validation[data_validation['labels'] == u_cl]
        np.save(predDir+'pred_validation__'+str(u_cl)+'.npy', tmp)


def load_Model():

    # load json and create model
    json_file = open(outputDir+'model_arch.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(outputDir+"model_weights.h5")
    print("Loaded model from disk")
    # don't need to compile if you only want to predict and not train/evaluate

    return loaded_model


def train_NN(parameters):

    """Do the actual training of your NN."""

    # split all datasets into training, test, and validation samples!
    splits = parameters['splits']
    seed = 0 # do this seeding more elegant in the future, see comment in split_TrainTestValidation function definition. However, need to ensure that data and weights use same seed!
    for u_cl in parameters['usedClasses']:
        split_TrainTestValidation(u_cl, splits['train'], splits['test'], splits['validation'], seed)
        split_TrainTestValidation(u_cl, splits['train'], splits['test'], splits['validation'], seed, '_weights')
        seed = seed+1

    # get data for Keras usage!
    data_train = make_DatasetUsableWithKeras(parameters['usedClasses'], 'train')
    data_test = make_DatasetUsableWithKeras(parameters['usedClasses'], 'test')
    data_validation = make_DatasetUsableWithKeras(parameters['usedClasses'], 'validation')

    # initialize your own custom history callback in which training set and validation set are evaluated after each epoch in the same way!
    customHistory = AdditionalValidationSets([
        (data_train['values'], data_train['encodedLabels'], data_train['weights'], 'train')#,
        #(data_validation['values'], data_validation['encodedLabels'], data_validation['weights'], 'valid')
    ])

    # initialize checkpointer callback
    filePathCheckPoints = outputDir+checkpointDir+'checkpoint-{epoch:03d}.h5'
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

    # predict labels of datasets!
    predict_Labels(parameters, model, data_train, data_test, data_validation)


def dump_ParametersIntoJsonFile(parameters):

    """Saves NN parameters to json file."""

    param_json = json.dumps(parameters)
    with open(outputDir+'parameters.json', 'w') as f:
        f.write(param_json)


if __name__ == '__main__':

    main()
