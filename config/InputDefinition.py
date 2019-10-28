# Specify which input variables you want to use (True/False)!
# Specify how many HOTVR and AK4 jets you want to use!

import os, sys
import numpy as np

# scheme: [variable name, estimated lower limit, estimated upper limit, boolean (use it for DNN?)]
# limit values are used for normalizing the input vector entries to interval [0,1]. If there is a case where an event has values outside the given limits, then there will be an entry with a value outside of the interval [0,1] but this shoud not matter that much. We just need this to numerically stabilize the DNN training.

template_event = [
    ["n_pv",0,100,False],
    ["met_px",-2000,2000,False],
    ["met_py",-2000,2000,False],
    ["met_pt",0,2000,True],
    ["met_phi",-4,4,True],
    ["ht_had",0,7000,True],
    ["ht_lep",0,2000,True],
    ["st",0,7000,True],
    ["n_hotvr",0,10,True],
    ["n_jets",0,20,True],
    ["n_btags",0,10,True]
]
template_hotvr = [
    ["toptagged",0,1,True],
    ["px",-2000,2000,False],
    ["py",-2000,2000,False],
    ["pz",-5000,5000,False],
    ["e",0,7000,False],
    ["pt",0,2000,True],
    ["eta",-5,5,True],
    ["phi",-4,4,True],
    ["v4mass",0,500,True],
    ["area",0,20,False],
    ["nsubjets",0,10,True],
    ["sub1_px",-2000,2000,False],
    ["sub1_py",-2000,2000,False],
    ["sub1_pz",-2000,2000,False],
    ["sub1_e",0,2000,False],
    ["sub1_pt",0,1000,False],
    ["sub1_eta",-5,5,False],
    ["sub1_phi",-4,4,False],
    ["sub1_v4mass",0,100,False],
    ["sub1_area",0,15,False],
    ["sub2_px",-2000,2000,False],
    ["sub2_py",-2000,2000,False],
    ["sub2_pz",-2000,2000,False],
    ["sub2_e",0,2000,False],
    ["sub2_pt",0,1000,False],
    ["sub2_eta",-5,5,False],
    ["sub2_phi",-4,4,False],
    ["sub2_v4mass",0,100,False],
    ["sub2_area",0,15,False],
    ["sub3_px",-2000,2000,False],
    ["sub3_py",-2000,2000,False],
    ["sub3_pz",-2000,2000,False],
    ["sub3_e",0,2000,False],
    ["sub3_pt",0,1000,False],
    ["sub3_eta",-5,5,False],
    ["sub3_phi",-4,4,False],
    ["sub3_v4mass",0,100,False],
    ["sub3_area",0,15,False],
    ["fpt",0,1,True],
    ["mpair",0,300,True],
    ["tau1",0,1,True],
    ["tau2",0,1,True],
    ["tau3",0,1,True],
    ["tau21",0,1,True],
    ["tau32",0,1,True]
]
template_jet = [
    ["btagged",0,1,False],
    ["DeepJet",0,1,True],
    ["px",-2000,2000,False],
    ["py",-2000,2000,False],
    ["pz",-2000,2000,False],
    ["e",0,7000,False],
    ["pt",0,3000,True],
    ["eta",-5,5,True],
    ["phi",-4,4,True],
    ["v4mass",0,300,True],
    ["area",0,2,False]
]
template_lepton = [
    ["px",-2000,2000,False],
    ["py",-2000,2000,False],
    ["pz",-2000,2000,False],
    ["e",0,7000,True],
    ["pt",0,2000,True],
    ["eta",-5,5,True],
    ["phi",-4,4,True],
    ["v4mass",-1,1,False], # measured in GeV, muon mass ca. 106 MeV, negative values possible
    ["reliso",0,0.2,True],
    ["charge",-1,1,True],
    ["dr_jet",0,6,True],
    ["dphi_jet",0,4,True],
    ["ptrel_jet",0,7000,True]
]


def compileInputList(n_hotvr, n_jets):

    """Returns an array of all DNN input variables together with their normalization parameters."""

    print "Number of HOTVR jets considered for input vector:  "+str(n_hotvr)
    print "Number of AK4   jets considered for input vector:  "+str(n_jets)
    n_hotvr = int(n_hotvr)
    n_jets = int(n_jets)
    inputList = []
    for var in template_event:
        if var[-1] == True:
            inputList.append(["DNN__"+var[0], float(var[1]), 1/float(var[2]-var[1])])
    for i in range(n_hotvr):
        for var in template_hotvr:
            if var[-1] == True:
                inputList.append(["DNN__hotvr"+str(i+1)+"_"+var[0], float(var[1]), 1/float(var[2]-var[1])])
    for i in range(n_jets):
        for var in template_jet:
            if var[-1] == True:
                inputList.append(["DNN__jet"+str(i+1)+"_"+var[0], float(var[1]), 1/float(var[2]-var[1])])
    for var in template_lepton:
        if var[-1] == True:
            inputList.append(["DNN__lepton_"+var[0], float(var[1]), 1/float(var[2]-var[1])])
    print "Length of input vector:                            "+str(len(inputList))
    return np.array(inputList)


def main(n_hotvr, n_jets):
    print compileInputList(n_hotvr, n_jets)


if __name__=="__main__":
    if len(sys.argv) < 3:
        print "Usage:\n" \
            "    %s <number of hotvr jets> <number of small jets>\n" \
            % os.path.basename(sys.argv[0])
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
