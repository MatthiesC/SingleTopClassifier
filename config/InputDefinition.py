# Specify which input variables you want to use (True/False)!
# Specify how many HOTVR and AK4 jets you want to use!

number_of_hotvr_jets = 2
number_of_ak4_jets = 5

import os, sys
import numpy as np

# scheme: [variable name, estimated lower limit, estimated upper limit, boolean (use it for DNN?)]
# limit values are used for normalizing the input vector entries to interval [0,1]. If there is a case where an event has values outside the given limits, then there will be an entry with a value outside of the interval [0,1] but this shoud not matter that much. We just need this to numerically stabilize the DNN training.

template_ttag = [
    ["channel",-3,3,True],
    ["year",-3,3,True],
    ["met_pt",0,2000,True],
    #["met_phi",-4,4,True],
    ["lepton_pt",0,2000,True],
    ["lepton_eta",-5,5,True],
    ["lepton_phi",-4,4,True],
    #["lepton_reliso",0,0.2,True],
    ["lepton_charge",-1,1,True],
    ["lepton_dr_nextjet",0,6,True],
    ["lepton_dphi_nextjet",0,4,True],
    ["lepton_deta_nextjet",0,6,True],
    ["lepton_ptrel_nextjet",0,800,True],
    ["lepton_dr_nextxjet",0,6,True],
    ["lepton_dphi_nextxjet",0,4,True],
    ["lepton_deta_nextxjet",0,6,True],
    ["lepton_ptrel_nextxjet",0,800,True],
    ["dphi_lepton_met",0,4,True],
    ["mtw",0,500,True],
    ["pseudotop_m",0,800,True],
    #["tjet_tau32",0,1,True],
    #["tjet_tau21",0,1,True],
    #["tjet_tau1",0,1,True],
    #["tjet_tau2",0,1,True],
    #["tjet_tau3",0,1,True],
    #["tjet_mjet",0,200,True],
    ["tjet_pt",0,1000,True],
    ["tjet_eta",-5,5,True],
    ["tjet_phi",-4,4,True],
    #["tjet_softdropmass",60,110,True],
    ["dr_tjet_lepton",0,6,True],
    ["dphi_tjet_met",0,4,True],
    ["ptbal_tjet_wboson",-2,2,True],
    ["ptbal_tjet_pseudotop",-2,2,True],
    ["n_xjets",0,10,True],
    ["ht_xjets",0,1000,True],
    #["xjet1pt_m",0,100,True],
    ["xjet1pt_pt",0,500,True],
    ["xjet1pt_eta",-5,5,True],
    ["xjet1pt_phi",-4,4,True],
    ["xjet1pt_deepjet",0,1,True],
    ["mass_xjet1pt_lepton",0,500,True],
    ["dr_xjet1pt_lepton",0,6,True],
    ["dr_xjet1pt_tjet",0,6,True],
    #["xjet1dj_m",0,100,True],
    ["xjet1dj_pt",0,500,True],
    ["xjet1dj_eta",-5,5,True],
    ["xjet1dj_phi",-4,4,True],
    ["xjet1dj_deepjet",0,1,True],
    ["mass_xjet1dj_lepton",0,500,True],
    ["dr_xjet1dj_lepton",0,6,True],
    ["dr_xjet1dj_tjet",0,6,True],
    #["xjet2dj_deepjet",0,1,True],
    ["ijet1dj_deepjet",0,1,True],
    #["ijet2dj_deepjet",0,1,True],
    ["dr_tjet_xjet",0,6,True],
    #["wnearestxjet_m",0,100,True],
    ["tnearestxjet_pt",0,500,True],
    ["tnearestxjet_eta",-5,5,True],
    ["tnearestxjet_phi",-4,4,True],
    ["tnearestxjet_deepjet",0,1,True],
]

template_wtag = [
    ["channel",-3,3,True],
    ["year",-3,3,True],
    ["met_pt",0,2000,True],
    #["met_phi",-4,4,True],
    ["lepton_pt",0,2000,True],
    ["lepton_eta",-5,5,True],
    ["lepton_phi",-4,4,True],
    #["lepton_reliso",0,0.2,True],
    ["lepton_charge",-1,1,True],
    ["lepton_dr_nextjet",0,6,True],
    ["lepton_dphi_nextjet",0,4,True],
    ["lepton_deta_nextjet",0,6,True],
    ["lepton_ptrel_nextjet",0,800,True],
    ["lepton_dr_nextxjet",0,6,True],
    ["lepton_dphi_nextxjet",0,4,True],
    ["lepton_deta_nextxjet",0,6,True],
    ["lepton_ptrel_nextxjet",0,800,True],
    ["dphi_lepton_met",0,4,True],
    ["mtw",0,500,True],
    ["pseudotop_m",0,800,True],
    #["wjet_tau32",0,1,True],
    #["wjet_tau21",0,1,True],
    #["wjet_tau1",0,1,True],
    #["wjet_tau2",0,1,True],
    #["wjet_tau3",0,1,True],
    #["wjet_mjet",0,200,True],
    ["wjet_pt",0,1000,True],
    ["wjet_eta",-5,5,True],
    ["wjet_phi",-4,4,True],
    #["wjet_softdropmass",60,110,True],
    ["dr_wjet_lepton",0,6,True],
    ["dphi_wjet_met",0,4,True],
    ["ptbal_wjet_wboson",-2,2,True],
    ["ptbal_wjet_pseudotop",-2,2,True],
    ["n_xjets",0,10,True],
    ["ht_xjets",0,1000,True],
    #["xjet1pt_m",0,100,True],
    ["xjet1pt_pt",0,500,True],
    ["xjet1pt_eta",-5,5,True],
    ["xjet1pt_phi",-4,4,True],
    ["xjet1pt_deepjet",0,1,True],
    ["mass_xjet1pt_lepton",0,500,True],
    ["dr_xjet1pt_lepton",0,6,True],
    ["dr_xjet1pt_wjet",0,6,True],
    #["xjet1dj_m",0,100,True],
    ["xjet1dj_pt",0,500,True],
    ["xjet1dj_eta",-5,5,True],
    ["xjet1dj_phi",-4,4,True],
    ["xjet1dj_deepjet",0,1,True],
    ["mass_xjet1dj_lepton",0,500,True],
    ["dr_xjet1dj_lepton",0,6,True],
    ["dr_xjet1dj_wjet",0,6,True],
    #["xjet2dj_deepjet",0,1,True],
    ["ijet1dj_deepjet",0,1,True],
    #["ijet2dj_deepjet",0,1,True],
    ["dr_wjet_xjet",0,6,True],
    #["wnearestxjet_m",0,100,True],
    ["wnearestxjet_pt",0,500,True],
    ["wnearestxjet_eta",-5,5,True],
    ["wnearestxjet_phi",-4,4,True],
    ["wnearestxjet_deepjet",0,1,True],
]


def compileInputList(region):

    """Returns an array of all DNN input variables together with their normalization parameters."""

    used_template = None
    if region=='wtag':
        used_template = template_wtag
    elif region=='ttag':
        used_template = template_ttag

    inputList = []
    for var in used_template:
        if var[-1] == True:
            inputList.append(["DNNinput_"+region+"__"+var[0], float(var[1]), 1/float(var[2]-var[1])])
    print("Length of input vector: "+str(len(inputList)))
    return inputList


def main(region):
    print(np.array(compileInputList(region)))


if __name__=="__main__":

    region = sys.argv[1]
    main(region)
