from config.InputDefinition import compileInputList
from config.SampleClasses import *

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json

import tensorflow as tf

from keras.utils import np_utils, plot_model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from CustomCallback import AdditionalValidationSets
from CustomLosses import categorical_focal_loss, binary_focal_loss
from Plotting import get_Parameters

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from datetime import datetime


### global variables

region = 'wtag'

# use a time tag to identify trained models later (for details, look up the parameters.json in the output dir)
timeTag = datetime.now()
timeTag = timeTag.strftime('%y-%m-%d-%H-%M-%S')
outputDir = './outputs/'+region+'_dnn_'+timeTag+'/'
checkpointDir = 'checkpoints/' #subfolder of outputDir



### end of global stuff




def main():

    """The main function."""

    print("Creating output directory:", outputDir)
    os.makedirs(outputDir+checkpointDir, exist_ok=True)
    os.makedirs(outputDir+'workdir/', exist_ok=True)

    parameters = {
        'region': region,
        'binary': False,
        'binary_signal': 'tW', # define background composition via usedClasses
        #'usedClasses': ['tW_signal', 'tW_bkg_TopToHadAndWToTau', 'tW_bkg_Else', 'TTbar', 'WJets', 'DYJets'],
        #'usedClasses': ['tW_signal', 'tW_bkg_TopToHadAndWToTau', 'tW_bkg_Else', 'TTbar'],
        #'usedClasses': ['tW_signal', 'TTbar'],
        #'usedClasses': ['tW_sig', 'tW_bkg', 'TTbar', 'WJets', 'DYJets', 'Diboson'],
        #'usedClasses': ['tW_sig', 'tW_bkg', 'TTbar', 'Electroweak'],
        'usedClasses': ['tW', 'TTbar', 'Electroweak'],
        'splits': { 'train': 0.7, 'test': 0.15, 'validation': 0.15 },
        'augmentation': True,
        'augment_weights_only': True, # 'False': Will take several minutes to augment data. Use 'True' for quick test runs
        'layers': [128, 128, 128, 128],
        'dropout': False,
        'dropout_rate': 0.1,
        'epochs': 100,
        'batch_size': 1024, #65536, #16384
        'learning_rate': 0.0001, #Adam default: 0.001
        'lr_decay': 0.00025,
        'regularizer': '', # either 'l1' or 'l2' or just ''
        'regularizer_rate': 0.01,
        'focal_loss': False,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        #'pt_preselection': (0, 99999), # interval of toptag pt
        'inputVariableNames': (np.array(compileInputList(region))[:,0]).tolist(),
        'inputVars': compileInputList(region) # for later use with lwtnn to compile variables.json
    }

    print("Using these input variables:", parameters.get('inputVariableNames'))

    dump_ParametersIntoJsonFile(parameters)
    train_NN(parameters)
    print("Done.")







def load_Numpy(processName, inputSuffix='_norm'):

    """Loads numpy file from work directory. Needs process name (TTbar, QCD, etc.) and file name suffix (default = '_norm')."""

    fileName = './workdir_'+region+'/'+processName+inputSuffix+'.npy'
    return np.load(fileName)


def split_TrainTestValidation(processName, pt_preselection, percentTrain, percentTest, percentValidation, shuffle_seed, inputSuffix='_norm'):

    """Splits a given numpy sample into training, test, and validation numpy files. Returns list of file names of train, test, and validation numpy files."""

    if percentTrain+percentTest+percentValidation > 1: sys.exit("Sum of percentages for training, test, and validation samples is greater than 1. Exit.")

    print("Load numpy file for process:", processName)

    loaded_numpy = load_Numpy(processName, inputSuffix)
    taggedjetpts = load_Numpy(processName, '_taggedjetpts')
    # perform preselection if wanted:
    if pt_preselection:
        loaded_numpy = loaded_numpy[(taggedjetpts.flatten() > pt_preselection[0]) & (taggedjetpts.flatten() < pt_preselection[1])]
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


def prepare_Dataset(parameters, sample_type, inputSuffix='_norm'): # sample_type = 'train', 'test', or 'validation'

    """Returns a gigantic pandas dataframe, containing all events to be trained/tested/validated on."""

    listOfDataFrames = []

    for u_cl in parameters.get('usedClasses'):

        fileName = outputDir+'workdir/'+u_cl+inputSuffix+'_'+sample_type+'.npy'
        dataArray = np.load(fileName)
        dataFrame = pd.DataFrame(data=dataArray, columns=parameters.get('inputVariableNames'))

        fileNameWeights = outputDir+'workdir/'+u_cl+'_weights_'+sample_type+'.npy'
        weightsArray = np.load(fileNameWeights)
        dataFrame['EventWeight'] = weightsArray

        dataFrame['Class'] = u_cl

        listOfDataFrames.append(dataFrame)

    completeDataFrame = pd.concat(listOfDataFrames, ignore_index=True, sort=False)

    return completeDataFrame


def make_DatasetUsableWithKeras(parameters, sample_type, inputSuffix='_norm'): # sample_type = 'train', 'test', or 'validation'

    """Returns a dictionary containing data values, string labels, and encoded labels which can directly be used with the model.fit() function of Keras."""

    # In the following, do everything as described here:
    # https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

    # load dataset
    data = prepare_Dataset(parameters, sample_type, inputSuffix).values
    data_values = data[:,0:-2] # input vectors for NN
    data_labels = data[:,-1] # classes associated to each event, given in string format
    data_weights = data[:,-2] # event weights

    # encode class values as integers
    data_encodedLabels = None
    if parameters.get('binary'):
        print('Binary mode chosen. Going to use this signal class:', parameters.get('binary_signal'))
        # data_encodedLabels = [0, 0, 1, 0, 1, 1, 0, ...] where 0 are samples which do not have label of binary signal
        data_encodedLabels = np.zeros(len(data_labels))
        data_encodedLabels[data_labels == parameters.get('binary_signal')] = 1
    else:
        print('Categorical mode chosen.')
        encoder = LabelEncoder()
        encoder.fit(data_labels)
        print('Used classes (order used for the encoding of output node numbers, i.e. node 0 corresponds to the first entry in this list here):')
        print(encoder.classes_)
        np.save(outputDir+'encoder_classes.npy', (np.array(encoder.classes_)).astype(str)) # conversion to str format needed to avoid 'Object arrays cannot be loaded when allow_pickle=False' error
        encoded_data_labels = encoder.transform(data_labels)

        # convert integers to dummy values (i.e. one-hot-encoded)
        data_encodedLabels = np_utils.to_categorical(encoded_data_labels)

    result = {"values": data_values, "labels": data_labels, "encodedLabels": data_encodedLabels, "weights": data_weights}

    return result


def augment_Dataset(parameters, data, sample_type):

    """Augments underrepresented classes in a given dataset up to sums of weights of the class with the highest sum of weights."""

    print("Augmenting "+str(sample_type)+" data...")

    augment_weights_only = parameters.get('augment_weights_only')
    binary = parameters.get('binary')
    signal = parameters.get('binary_signal')

    sums_of_weights = dict()
    sums_of_weights_aug = dict()
    max_sum = 0.
    max_ucl = ''
    sum_bkg = 0 # for binary case

    for u_cl in parameters.get('usedClasses'):
        tmp = data['weights'][data['labels'] == u_cl]
        sums_of_weights[u_cl] = tmp.sum()
        if tmp.sum() > max_sum:
            max_sum = tmp.sum()
            max_ucl = u_cl
        if binary:
            if u_cl != signal:
                sum_bkg += tmp.sum()

    # for binary case:
    augment_signal = False
    if sum_bkg > sums_of_weights.get(signal):
        augment_signal = True

    data_aug = dict()

    data_aug_values = np.empty([0, len(parameters.get('inputVariableNames'))])
    data_aug_labels = np.array(list())
    data_aug_weights = np.array(list())
    data_aug_encodedLabels = None
    if binary:
        data_aug_encodedLabels = np.array(list())
    else:
        data_aug_encodedLabels = np.empty([0, len(parameters.get('usedClasses'))])

    if augment_weights_only:

        for u_cl in parameters.get('usedClasses'):

            global_scale_factor = 1
            if binary:
                if augment_signal:
                    if u_cl == signal:
                        global_scale_factor = sum_bkg/sums_of_weights.get(signal)
                else:
                    if u_cl != signal:
                        global_scale_factor = sums_of_weights.get(signal)/sum_bkg
            else:
                global_scale_factor = max_sum/sums_of_weights[u_cl]

            tmp = data['weights'][data['labels'] == u_cl]
            tmp = tmp*global_scale_factor
            data_aug_weights = np.append(data_aug_weights, tmp)

            data_aug_values = np.concatenate((data_aug_values, data['values'][data['labels'] == u_cl]))
            data_aug_labels = np.append(data_aug_labels, data['labels'][data['labels'] == u_cl])
            if binary:
                data_aug_encodedLabels = np.append(data_aug_encodedLabels, data['encodedLabels'][data['labels'] == u_cl])
            else:
                data_aug_encodedLabels = np.concatenate((data_aug_encodedLabels, data['encodedLabels'][data['labels'] == u_cl]))

        data_aug['values'] = data_aug_values
        data_aug['labels'] = data_aug_labels
        data_aug['encodedLabels'] = data_aug_encodedLabels
        data_aug['weights'] = data_aug_weights

    else: # this will take some minutes! For testing, use augment_weights_only=True

        for u_cl in parameters.get('usedClasses'):

            global_scale_factor = max_sum/sums_of_weights[u_cl]

            data_aug_values_ucl = np.empty([0, len(parameters.get('inputVariableNames'))])
            data_aug_labels_ucl = np.array(list())
            data_aug_encodedLabels_ucl = np.empty([0, len(parameters.get('usedClasses'))])
            data_aug_weights_ucl = np.array(list())

            cardinality_ucl = len(data['values'][data['labels'] == u_cl])
            overhang = int((global_scale_factor - int(global_scale_factor))*cardinality_ucl)

            for i in range(int(global_scale_factor)):
                data_aug_values_ucl = np.concatenate((data_aug_values_ucl, data['values'][data['labels'] == u_cl]))
                data_aug_labels_ucl = np.append(data_aug_labels_ucl, data['labels'][data['labels'] == u_cl])
                data_aug_encodedLabels_ucl = np.concatenate((data_aug_encodedLabels_ucl, data['encodedLabels'][data['labels'] == u_cl]))
                data_aug_weights_ucl = np.append(data_aug_weights_ucl, data['weights'][data['labels'] == u_cl])

            data_aug_values_ucl = np.concatenate((data_aug_values_ucl, data['values'][data['labels'] == u_cl][0:overhang]))
            data_aug_labels_ucl = np.append(data_aug_labels_ucl, data['labels'][data['labels'] == u_cl][0:overhang])
            data_aug_encodedLabels_ucl = np.concatenate((data_aug_encodedLabels_ucl, data['encodedLabels'][data['labels'] == u_cl][0:overhang]))
            data_aug_weights_ucl = np.append(data_aug_weights_ucl, data['weights'][data['labels'] == u_cl][0:overhang])

            data_aug_values = np.concatenate((data_aug_values, data_aug_values_ucl))
            data_aug_labels = np.append(data_aug_labels, data_aug_labels_ucl)
            data_aug_encodedLabels = np.concatenate((data_aug_encodedLabels, data_aug_encodedLabels_ucl))
            data_aug_weights = np.append(data_aug_weights, data_aug_weights_ucl)

        data_aug['values'] = data_aug_values
        data_aug['labels'] = data_aug_labels
        data_aug['encodedLabels'] = data_aug_encodedLabels
        data_aug['weights'] = data_aug_weights

    # sanity check whether sum of weights for all classes now approximately equal:
    print("Sanity check after data augmentation:")
    for u_cl in parameters.get('usedClasses'):
        tmp = data_aug['weights'][data_aug['labels'] == u_cl]
        sums_of_weights_aug[u_cl] = tmp.sum()
        print("Sum of weights for non-augm. class "+str(u_cl)+": ", str(sums_of_weights[u_cl]))
        print("Sum of weights for augmented class "+str(u_cl)+": ", str(sums_of_weights_aug[u_cl]))
        print("Number of MC events for class "+str(u_cl)+": ", str(len(tmp)))

    return data_aug


def define_NetworkArchitecture(parameters):

    """Define the NN architecture."""

    layers = parameters.get('layers')

    my_kernel_regularizer = None
    if parameters.get('regularizer') == 'l1':
        my_kernel_regularizer = regularizers.l1(parameters.get('regularizer_rate'))
    elif parameters.get('regularizer') == 'l2':
        my_kernel_regularizer = regularizers.l2(parameters.get('regularizer_rate'))

    model = Sequential()
    model.add(Dense(layers[0], input_dim=len(parameters.get('inputVariableNames')), kernel_regularizer=my_kernel_regularizer, activation='relu'))
    if parameters.get('dropout'): model.add(Dropout(parameters.get('dropout_rate')))
    for i in range(len(layers)):
        if i == 0: continue
        model.add(Dense(layers[i], activation='relu', kernel_regularizer=my_kernel_regularizer))
        if parameters.get('dropout'): model.add(Dropout(parameters.get('dropout_rate')))
    if parameters.get('binary'):
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(len(parameters.get('usedClasses')), activation='softmax'))

    my_optimizer = optimizers.Adam(lr=parameters.get('learning_rate'), decay=parameters.get('lr_decay'))
    my_metrics = None
    my_loss = None
    if parameters.get('binary'):
        my_metrics = [metrics.binary_accuracy]
        if parameters.get('focal_loss'):
            my_loss = [binary_focal_loss(alpha=parameters.get('focal_alpha'), gamma=parameters.get('focal_gamma'))]
        else:
            my_loss = 'binary_crossentropy'
    else:
        my_metrics = [metrics.categorical_accuracy]
        if parameters.get('focal_loss'):
            my_loss = [categorical_focal_loss(alpha=parameters.get('focal_alpha'), gamma=parameters.get('focal_gamma'))]
        else:
            my_loss = 'categorical_crossentropy'

    model.compile(loss=my_loss, optimizer=my_optimizer, metrics=my_metrics)

    # save network architecture to disk before training begins!
    architecture = model.to_json()
    with open(outputDir+'model_arch.json', 'w') as f:
        f.write(architecture)
    plot_model(model, to_file=outputDir+'model_arch.png')

    return model


def predict_Labels(parameters, model, data_train, data_test, data_validation, augmented=False):

    """Predict labels of train/test/validation datasets using the trained model."""

    predDir = outputDir+'predictions/'
    os.makedirs(predDir, exist_ok=True)

    augPostFix = ''
    if augmented: augPostFix = '_aug'

    print("Predicting labels of training dataset...")
    pred_train = model.predict(data_train['values'])
    np.save(predDir+'pred'+augPostFix+'_train.npy', pred_train)
    for u_cl in parameters.get('usedClasses'):
        tmp = pred_train[data_train['labels'] == u_cl]
        np.save(predDir+'pred'+augPostFix+'_train__'+str(u_cl)+'.npy', tmp)

    print("Predicting labels of test dataset...")
    pred_test = model.predict(data_test['values'])
    np.save(predDir+'pred'+augPostFix+'_test.npy', pred_test)
    for u_cl in parameters.get('usedClasses'):
        tmp = pred_test[data_test['labels'] == u_cl]
        np.save(predDir+'pred'+augPostFix+'_test__'+str(u_cl)+'.npy', tmp)

    print("Predicting labels of validation dataset...")
    pred_validation = model.predict(data_validation['values'])
    np.save(predDir+'pred'+augPostFix+'_validation.npy', pred_validation)
    for u_cl in parameters.get('usedClasses'):
        tmp = pred_validation[data_validation['labels'] == u_cl]
        np.save(predDir+'pred'+augPostFix+'_validation__'+str(u_cl)+'.npy', tmp)


def load_Model():

    """Load previously trained model from disk."""

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
    splits = parameters.get('splits')
    seed = 0 # do this seeding more elegant in the future, see comment in split_TrainTestValidation function definition. However, need to ensure that data and weights use same seed!
    for u_cl in parameters.get('usedClasses'):
        split_TrainTestValidation(u_cl, parameters.get('pt_preselection'), splits['train'], splits['test'], splits['validation'], seed)
        split_TrainTestValidation(u_cl, parameters.get('pt_preselection'), splits['train'], splits['test'], splits['validation'], seed, '_weights')
        seed = seed+1

    # get data for Keras usage!
    data_train_raw = make_DatasetUsableWithKeras(parameters, 'train')
    data_test_raw = make_DatasetUsableWithKeras(parameters, 'test')
    data_validation_raw = make_DatasetUsableWithKeras(parameters, 'validation')

    # augment all datasets, such that they have equal sum of weights inside loss function!
    if parameters.get('augmentation'):
        data_train = augment_Dataset(parameters, data_train_raw, 'train')
        data_test = augment_Dataset(parameters, data_test_raw, 'test')
        data_validation = augment_Dataset(parameters, data_validation_raw, 'validation')
    else:
        data_train, data_test, data_validation = data_train_raw, data_test_raw, data_validation_raw

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
    with tf.device('/gpu:0'):
        history = model.fit(data_train['values'], data_train['encodedLabels'], sample_weight=data_train['weights'], epochs=parameters.get('epochs'), batch_size=parameters.get('batch_size'), shuffle=True, validation_data=(data_validation['values'], data_validation['encodedLabels'], data_validation['weights']), callbacks=[customHistory, checkpointer])
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
    predict_Labels(parameters, model, data_train_raw, data_test_raw, data_validation_raw) # raw datasets
    predict_Labels(parameters, model, data_train, data_test, data_validation, True) # augmented datasets


def dump_ParametersIntoJsonFile(parameters):

    """Saves NN parameters to json file."""

    param_json = json.dumps(parameters)
    with open(outputDir+'parameters.json', 'w') as f:
        f.write(param_json)


if __name__ == '__main__':

    main()
