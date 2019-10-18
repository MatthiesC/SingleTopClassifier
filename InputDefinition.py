# Specify which input variables you want to use (True/False)!
# Specify how many HOTVR and AK4 jets you want to use!

import os, sys

template_event = [
    ["n_pv",False],
    ["met_px",False],
    ["met_py",False],
    ["met_pt",True],
    ["met_phi",True],
    ["ht_had",True],
    ["ht_lep",True],
    ["st",True],
    ["n_hotvr",True],
    ["n_jets",True],
    ["n_btags",True]
]
template_hotvr = [
    ["toptagged",True],
    ["px",False],
    ["py",False],
    ["pz",False],
    ["e",False],
    ["pt",True],
    ["eta",True],
    ["phi",True],
    ["v4mass",True],
    ["area",False],
    ["nsubjets",True],
    ["sub1_px",False],
    ["sub1_py",False],
    ["sub1_pz",False],
    ["sub1_e",False],
    ["sub1_pt",False],
    ["sub1_eta",False],
    ["sub1_phi",False],
    ["sub1_v4mass",False],
    ["sub1_area",False],
    ["sub2_px",False],
    ["sub2_py",False],
    ["sub2_pz",False],
    ["sub2_e",False],
    ["sub2_pt",False],
    ["sub2_eta",False],
    ["sub2_phi",False],
    ["sub2_v4mass",False],
    ["sub2_area",False],
    ["sub3_px",False],
    ["sub3_py",False],
    ["sub3_pz",False],
    ["sub3_e",False],
    ["sub3_pt",False],
    ["sub3_eta",False],
    ["sub3_phi",False],
    ["sub3_v4mass",False],
    ["sub3_area",False],
    ["fpt",True],
    ["mpair",True],
    ["tau1",True],
    ["tau2",True],
    ["tau3",True],
    ["tau21",True],
    ["tau32",True]
]
template_jet = [
    ["btagged",True],
    ["DeepJet",False],
    ["px",False],
    ["py",False],
    ["pz",False],
    ["e",False],
    ["pt",True],
    ["eta",True],
    ["phi",True],
    ["v4mass",True],
    ["area",False]
]
template_lepton = [
    ["px",False],
    ["py",False],
    ["pz",False],
    ["e",True],
    ["pt",True],
    ["eta",True],
    ["phi",True],
    ["v4mass",False],
    ["reliso",True],
    ["charge",True],
    ["dr_jet",True],
    ["dphi_jet",True],
    ["ptrel_jet",True]
]


def compileInputList(n_hotvr, n_jets):
    print "Number of HOTVR jets considered for input vector:  "+str(n_hotvr)
    print "Number of AK4   jets considered for input vector:  "+str(n_jets)
    n_hotvr = int(n_hotvr)
    n_jets = int(n_jets)
    inputList = []
    for var in template_event:
        if var[1] == True:
            inputList.append("DNN__"+var[0])
    for i in range(n_hotvr):
        for var in template_hotvr:
            if var[1] == True:
                inputList.append("DNN__hotvr"+str(i)+"_"+var[0])
    for i in range(n_jets):
        for var in template_jet:
            if var[1] == True:
                inputList.append("DNN__jet"+str(i)+"_"+var[0])
    for var in template_lepton:
        if var[1] == True:
            inputList.append("DNN__lepton_"+var[0])
    print "Length of input vector:                            "+str(len(inputList))
    return inputList


def main(n_hotvr, n_jets):
    print compileInputList(n_hotvr, n_jets)


if __name__=="__main__":
    if len(sys.argv) < 3:
        print "Usage:\n" \
            "    %s <number of hotvr jets> <number of small jets>\n" \
            % os.path.basename(sys.argv[0])
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
