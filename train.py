import os

from config.InputDefinition import compileInputList
from config.SampleClasses import dict_Classes, fileNamePrefix_MC, fileNamePrefix_DATA

from root_numpy import root2array
import numpy as np
import pandas

from keras.models import Sequential
from keras.layers import Dense


def get_InputVariableNames(n_hotvr=2, n_jets=4):

    """Compiles a list of the DNN input variables based on InputDefinition."""

    inputList = compileInputList(n_hotvr, n_jets)

    return inputList


def read_InputVariables(rootFileName, inputArray, treeName='AnalysisTree'):

    """Reads all DNN input variables from a given ROOT file and translates the input into DataFrame format."""

    inputNames = inputArray[:,0] # get only the string names
    numpyArray = root2array(rootFileName, treename=treeName, branches=inputNames)
    dataFrame = pandas.DataFrame(numpyArray)

    return dataFrame


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


def define_KerasModel(inputArray):

    """Defines the DNN model."""

    model = Sequential()
    model.add(Dense(512, input_dim=len(inputArray), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def main():

    inputVariables = get_InputVariableNames()
    model = define_KerasModel(inputVariables)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.fit(X, y, epochs=10, batch_size=1024)

    dataFrame = read_InputVariables(fileNamePrefix_MC+dict_Classes['QCD']['File'], inputVariables)
    create_Path('workdir')
    np.save('workdir/QCD.npy', dataFrame)
    loaded_input = np.load('workdir/QCD.npy')
    print "Input vector:"
    print loaded_input
    norm_input = normalize_InputVectorEntries(loaded_input, inputVariables)
    np.save('workdir/QCD_norm.npy', norm_input)
    print "Normalized input vector:"
    print norm_input


if __name__=="__main__":

    main()
