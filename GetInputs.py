import os

from config.InputDefinition import compileInputList
from config.SampleClasses import dict_Classes, fileNamePrefix_MC, fileNamePrefix_DATA

import ROOT
import numpy as np
import pandas


def get_InputVariableParameters():

    """Compiles a list of the DNN input variables based on InputDefinition."""

    inputList = np.array(compileInputList())

    return inputList


def read_InputVariables(rootFileName, inputArray, treeName='AnalysisTree'):

    """Reads all DNN input variables from a given ROOT file and translates the input into DataFrame format."""

    inputNames = inputArray[:,0] # get only the string names
    rootFile = ROOT.TFile(rootFileName)
    rootTree = rootFile.Get(treeName)
    numpyArray = rootTree.AsMatrix(columns=inputNames.tolist())
    dataFrame = pandas.DataFrame(numpyArray.astype(float))

    return dataFrame


def read_SpecificVariable(variableName, rootFileName, treeName='AnalysisTree'):

    """Reads a specific variable from a given ROOT file and translates it into numpy format. Be careful that it is read in the same order as all input variables."""

    rootFile = ROOT.TFile(rootFileName)
    rootTree = rootFile.Get(treeName)
    numpyArray = rootTree.AsMatrix(columns=[variableName])

    return numpyArray.astype(float)


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
        print("The path '%s' already exists, not creating it." % (path))
    else:
        os.makedirs(path)
        print("Created path '%s'" % (path))


def save_NumpyFiles(processName, is_mc, verbose=False, workdir='workdir'):

    """Save input vectors and event weights as numpy files in workdir. Specify process name (TTbar, WJets, etc.) as in SampleClasses.py"""

    create_Path(workdir)

    print("Saving numpy files for process:", processName)

    inputVariables = get_InputVariableParameters()

    dataFrame = None
    if is_mc:
        dataFrame = read_InputVariables(fileNamePrefix_MC+dict_Classes[processName]['File'], inputVariables)
    else:
        dataFrame = read_InputVariables(fileNamePrefix_DATA+dict_Classes[processName]['File'], inputVariables)

    path = workdir+'/'+processName+'.npy'
    np.save(path, dataFrame)
    loaded_input = np.load(path)
    print("Number of events:", len(loaded_input))
    if verbose: print("Input vector:\n", loaded_input)
    norm_input = normalize_InputVectorEntries(loaded_input, inputVariables)
    path = workdir+'/'+processName+'_norm.npy'
    np.save(path, norm_input)
    if verbose: print("Normalized input vector:", norm_input)
    if is_mc:
        weights = read_SpecificVariable('DNN_EventWeight', fileNamePrefix_MC+dict_Classes[processName]['File'])
        toptagpts = read_SpecificVariable('DNN_TopTagPt', fileNamePrefix_MC+dict_Classes[processName]['File'])
        if verbose:
            print("Event weight vector:", weights)
            print("Top-tag pT vector:", toptagpts)
        path = workdir+'/'+processName
        np.save(path+'_weights.npy', weights)
        np.save(path+'_toptagpts.npy', toptagpts)


def setup_Inputs():

    """Main function of this script. Converts UHH2 ntuples into numpy format, including normalizing the input variables and also saving event weights into separate numpy files."""

    processes = []
    for proc in dict_Classes.keys():
        if dict_Classes[proc]["Use"] == True:
            processes.append(proc)
    print("Working on theses processes:", processes)

    for p in processes:
        save_NumpyFiles(p, True)


if __name__=="__main__":

    setup_Inputs()
