# Specify which input variables you want to use (True/False)!
# Specify how many HOTVR and AK4 jets you want to use!

number_of_hotvr_jets = 2
number_of_ak4_jets = 5

import os, sys
import numpy as np

# scheme: [variable name, estimated lower limit, estimated upper limit, boolean (use it for DNN?)]
# limit values are used for normalizing the input vector entries to interval [0,1]. If there is a case where an event has values outside the given limits, then there will be an entry with a value outside of the interval [0,1] but this shoud not matter that much. We just need this to numerically stabilize the DNN training.

template_event = [
    ["n_pv",0,100,False],
    ["met_px",-2000,2000,False],
    ["met_py",-2000,2000,False],
    ["met_pt",0,2000,True],
    ["met_phi",-4,4,False],
    ["ht_had",0,7000,False],
    ["ht_lep",0,2000,False],
    ["st",0,7000,False],
    ["n_hotvr",0,10,False],
    ["n_jets",0,20,False],
    ["n_btags",0,10,False]
]
template_hotvr = [
    ["toptagged",0,1,False],
    ["px",-2000,2000,False],
    ["py",-2000,2000,False],
    ["pz",-5000,5000,False],
    ["e",0,7000,False],
    ["pt",0,2000,False],
    ["eta",-5,5,False],
    ["phi",-4,4,False],
    ["v4mass",0,500,False],
    ["area",0,20,False],
    ["nsubjets",0,10,False],
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
    ["fpt",0,1,False],
    ["mpair",0,300,False],
    ["tau1",0,1,False],
    ["tau2",0,1,False],
    ["tau3",0,1,False],
    ["tau21",0,1,False],
    ["tau32",0,1,False]
]
template_jet = [
    ["btagged",0,1,False],
    ["DeepJet",0,1,False],
    ["px",-2000,2000,False],
    ["py",-2000,2000,False],
    ["pz",-2000,2000,False],
    ["e",0,7000,False],
    ["pt",0,3000,False],
    ["eta",-5,5,False],
    ["phi",-4,4,False],
    ["v4mass",0,300,False],
    ["area",0,2,False]
]
template_lepton = [
    ["px",-2000,2000,False],
    ["py",-2000,2000,False],
    ["pz",-2000,2000,False],
    ["e",0,7000,False],
    ["pt",0,2000,True],
    ["eta",-5,5,True],
    ["phi",-4,4,False],
    ["v4mass",-1,1,False], # measured in GeV, muon mass ca. 106 MeV, negative values possible
    ["reliso",0,0.2,True],
    ["charge",-1,1,True],
    ["dr_jet",0,6,True],
    ["dphi_jet",0,4,False],
    ["ptrel_jet",0,800,True]
]
template_custom = [
    ["tjet_tau32",0,1,True],
    ["tjet_tau21",0,1,True],
    ["tjet_tau1",0,1,True],
    ["tjet_tau2",0,1,True],
    ["tjet_tau3",0,1,True],
    ["tjet_fpt",0,1,True],
    ["tjet_mjet",0,500,True],
    ["tjet_mij",0,300,True],
    ["dR_tl",0,6,True],
    ["dPhi_tm",0,4,True],
    ["dPhi_lm",0,4,True],
    ["pTbal_wt",-2,2,True],
    ["pTbal_tlepthad",-2,2,True],
    ["m_top",0,500,True],
    ["mt_w",0,300,True],
    ["n_xjets",0,10,True],
    ["ht_xjets",0,1000,True],
    ["xjet1_m",0,100,True],
    ["xjet1_pt",0,1000,True],
    ["xjet1_eta",-5,5,True],
    ["xjet1_deepjet",0,1,True],
    ["mass_xjet1_lep",0,500,True],
    ["dr_xjet1_lep",0,6,True]
]


def compileInputList(n_hotvr=number_of_hotvr_jets, n_jets=number_of_ak4_jets):

    """Returns an array of all DNN input variables together with their normalization parameters."""

    print("Number of HOTVR jets considered for input vector:  "+str(n_hotvr))
    print("Number of AK4   jets considered for input vector:  "+str(n_jets))
    n_hotvr = int(n_hotvr)
    n_jets = int(n_jets)
    inputList = []
    for var in template_event:
        if var[-1] == True:
            inputList.append(["DNN__event_"+var[0], float(var[1]), 1/float(var[2]-var[1])])
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
    for var in template_custom:
        if var[-1] == True:
            inputList.append(["DNN__custom_"+var[0], float(var[1]), 1/float(var[2]-var[1])])
    print("Length of input vector:                            "+str(len(inputList)))
    return np.array(inputList)


def main(n_hotvr, n_jets):
    print(compileInputList(n_hotvr, n_jets))


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("Usage:\n" \
            "    %s <number of hotvr jets> <number of small jets>\n" \
            % os.path.basename(sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
