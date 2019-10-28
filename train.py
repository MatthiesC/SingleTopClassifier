import os

from config.InputDefinition import compileInputList
from config.SampleClasses import dict_Classes, fileNamePrefix_MC, fileNamePrefix_DATA

from root_numpy import root2array
import numpy as np
import pandas

from keras.models import Sequential
from keras.layers import Dense


def get_InputVariableParameters():

    """Compiles a list of the DNN input variables based on InputDefinition."""

    inputList = compileInputList()

    return inputList


def read_InputVariables(rootFileName, inputArray, treeName='AnalysisTree'):

    """Reads all DNN input variables from a given ROOT file and translates the input into DataFrame format."""

    inputNames = inputArray[:,0] # get only the string names
    numpyArray = root2array(rootFileName, treename=treeName, branches=inputNames)
    dataFrame = pandas.DataFrame(numpyArray)

    return dataFrame


def read_EventWeights(rootFileName, treeName='AnalysisTree'):

    """Reads the event weights from a given ROOT file and translates it into numpy format. Be careful that it is read in the same order as the input variables."""

    weightsName = 'DNN_EventWeight'
    numpyArray = root2array(rootFileName, treename=treeName, branches=weightsName)

    return numpyArray


def normalize_InputVectorEntries(numpyInput, inputParameters):

    """Takes the input vector in numpy format and returns a new array with normalized input variables based on normalization parameters."""

    normalizedInput = numpyInput
    for entry in normalizedInput:
        for i in range(len(inputParameters)):
            entry[i] = (float(entry[i])-float(inputParameters[i,1]))*float(inputParameters[i,2])

    return normalizedInput


def create_Path(path):

    """Creates path given via argument. Useful to create workdir if not yet existing."""

    if os.path.isdir(path):
        print "The path '%s' already exists, not creating it." % (path)
    else:
        os.makedirs(path)
        print "Created path '%s'" % (path)


def save_NumpyFiles(processName, is_mc, verbose=False, workdir='workdir'):

    """Save input vectors and event weights as numpy files in workdir. Specify process name (TTbar, WJets, etc.) as in SampleClasses.py"""

    create_Path(workdir)

    print "Saving numpy files for process:", processName

    inputVariables = get_InputVariableParameters()

    dataFrame = None
    if is_mc:
        dataFrame = read_InputVariables(fileNamePrefix_MC+dict_Classes[processName]['File'], inputVariables)
    else:
        dataFrame = read_InputVariables(fileNamePrefix_DATA+dict_Classes[processName]['File'], inputVariables)

    path = workdir+'/'+processName+'.npy'
    np.save(path, dataFrame)
    loaded_input = np.load(path)
    print "Number of events:", len(loaded_input)
    if verbose: print "Input vector:"
    if verbose: print loaded_input
    norm_input = normalize_InputVectorEntries(loaded_input, inputVariables)
    path = workdir+'/'+processName+'_norm.npy'
    np.save(path, norm_input)
    if verbose: print 'Normalized input vector:'
    if verbose: print norm_input
    if is_mc:
        weights = read_EventWeights(fileNamePrefix_MC+dict_Classes['QCD']['File'])
        if verbose: print 'Event weight vector:'
        if verbose: print weights
        path = workdir+'/'+processName+'_weights.npy'
        np.save(path, weights)


def define_KerasModel(inputArray):

    """Defines the DNN model."""

    model = Sequential()
    model.add(Dense(512, input_dim=len(inputArray), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def setup_Inputs():

    """Main function of this script. Converts UHH2 ntuples into numpy format, including normalizing the input variables and also saving event weights into separate numpy files."""

    #inputVariables = get_InputVariableParameters()
    #model = define_KerasModel(inputVariables)
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(X, y, epochs=10, batch_size=1024)

    processes = []
    for proc in dict_Classes.keys():
        if dict_Classes[proc]["Use"] == True:
            processes.append(proc)
    print "Working on theses processes:",processes

    for p in processes:
        save_NumpyFiles(p, True)


if __name__=="__main__":

    setup_Inputs()
