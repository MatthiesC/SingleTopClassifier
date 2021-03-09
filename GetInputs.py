import os

from config.InputDefinition import compileInputList
from config.SampleClasses import dict_Classes, fileNamePrefix_MC, fileNamePrefix_DATA

import ROOT
import numpy as np
import pandas


def get_InputVariableParameters(region):

    """Compiles a list of the DNN input variables based on InputDefinition."""

    inputList = np.array(compileInputList(region))

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


def save_NumpyFiles(processName, is_mc, region, verbose=False, workdir_='workdir'):

    """Save input vectors and event weights as numpy files in workdir. Specify process name (TTbar, WJets, etc.) as in SampleClasses.py"""

    workdir = workdir_+'_'+region

    create_Path(workdir)

    print("Saving numpy files for region/process:", region, processName)

    inputVariables = get_InputVariableParameters(region)

    dataFrame = None
    #dataFrame_ele = None
    #dataFrame_muo = None
    #if is_mc:
    dataFrame = read_InputVariables(fileNamePrefix_MC.replace('samples/', 'samples/run2/')+dict_Classes[processName]['File'], inputVariables)
    #dataFrame_ele = read_InputVariables(fileNamePrefix_MC.replace('samples/', 'samples/ele/')+dict_Classes[processName]['File'], inputVariables)
    #dataFrame_muo = read_InputVariables(fileNamePrefix_MC.replace('samples/', 'samples/muo/')+dict_Classes[processName]['File'], inputVariables)
    # else:
    # dataFrame = read_InputVariables(fileNamePrefix_DATA+dict_Classes[processName]['File'], inputVariables)

    #dataFrame = pandas.concat([dataFrame_ele, dataFrame_muo], ignore_index=True, sort=False)

    which_region = read_SpecificVariable('which_region', fileNamePrefix_MC.replace('samples/', 'samples/run2/')+dict_Classes[processName]['File']) # 1 = ttag region, 2 = wtag region
    which_region = which_region.flatten()
    region_value = 0
    if region=='ttag':
        region_value = 1
    elif region=='wtag':
        region_value = 2
    which_region_boolean = which_region == region_value

    path = workdir+'/'+processName+'.npy'
    np.save(path, dataFrame)
    loaded_input_nottrimmed = np.load(path)
    loaded_input = loaded_input_nottrimmed[which_region_boolean]
    print("Number of events:", len(loaded_input))
    if verbose: print("Input vector:\n", loaded_input)
    norm_input = normalize_InputVectorEntries(loaded_input, inputVariables)
    path = workdir+'/'+processName+'_norm.npy'
    np.save(path, norm_input)
    if verbose: print("Normalized input vector:", norm_input)
    #if is_mc:
    weights = read_SpecificVariable('DNNinfo_event_weight', fileNamePrefix_MC.replace('samples/', 'samples/run2/')+dict_Classes[processName]['File'])
    weights_trimmed = weights[which_region_boolean]
    #weights_ele = read_SpecificVariable('DNNinfo_event_weight', fileNamePrefix_MC.replace('samples/', 'samples/ele/')+dict_Classes[processName]['File'])
    #weights_muo = read_SpecificVariable('DNNinfo_event_weight', fileNamePrefix_MC.replace('samples/', 'samples/muo/')+dict_Classes[processName]['File'])
    #wtagpts = read_SpecificVariable('DNNinfo_wjet_pt', fileNamePrefix_MC.replace('samples/', 'samples/run2/')+dict_Classes[processName]['File'])
    #wtagpts_ele = read_SpecificVariable('DNNinfo_wjet_pt', fileNamePrefix_MC.replace('samples/', 'samples/ele/')+dict_Classes[processName]['File'])
    #wtagpts_muo = read_SpecificVariable('DNNinfo_wjet_pt', fileNamePrefix_MC.replace('samples/', 'samples/muo/')+dict_Classes[processName]['File'])
    taggedjetpts = read_SpecificVariable('DNNinfo_'+region.replace('tag', 'jet')+'_pt', fileNamePrefix_MC.replace('samples/', 'samples/run2/')+dict_Classes[processName]['File'])
    taggedjetpts_trimmed = taggedjetpts[which_region_boolean]
    #weights = np.concatenate((weights_ele, weights_muo))
    #wtagpts = np.concatenate((wtagpts_ele, wtagpts_muo))
    if verbose:
        print("Event weight vector:", weights_trimmed)
        #print("W-tag pT vector:", wtagpts)
        print(region+" pT vector:", taggedjetpts_trimmed)
    path = workdir+'/'+processName
    np.save(path+'_weights.npy', weights_trimmed)
    np.save(path+'_taggedjetpts.npy', taggedjetpts_trimmed)
    #np.save(path+'_ttagpts.npy', ttagpts)
    #np.save(path+'_whichregion.npy', which_region)


def setup_Inputs():

    """Main function of this script. Converts UHH2 ntuples into numpy format, including normalizing the input variables and also saving event weights into separate numpy files."""

    processes = []
    for proc in dict_Classes.keys():
        if dict_Classes[proc]["Use"] == True:
            processes.append(proc)
    print("Working on theses processes:", processes)

    for p in processes:
        save_NumpyFiles(p, True, 'ttag')
        save_NumpyFiles(p, True, 'wtag')


if __name__=="__main__":

    setup_Inputs()
