fileNamePrefix_MC = "samples/uhh2.AnalysisModuleRunner.MC."
fileNamePrefix_DATA = "samples/uhh2.AnalysisModuleRunner.DATA."

dict_Classes = {
    "tW":{
        "Use":True,
        "File":"ST_tW_DR_NoFullyHadronic.root",
        "color": "red",
        "latex": r"tW"
    },
    "tW_sig":{
        "Use":True,
        "File":"ST_tW_DR_NoFullyHadronic_Sig.root",
        "color": "red",
        "latex": r"tW signal"
    },
    "tW_bkg":{
        "Use":True,
        "File":"ST_tW_DR_NoFullyHadronic_Bkg.root",
        "color": "magenta",
        "latex": r"tW bkg."
    },
    "TTbar":{
        "Use":True,
        "File":"TTbar.root",
        "color": "orange",
        "latex": r"$\mathrm{t}\bar{\mathrm{t}}$"
    },
    "singletop":{
        "Use":True,
        "File":"ST_otherChannels.root",
        "color": "gold",
        "latex": r"Other single t"
    },
    "WJets":{
        "Use":True,
        "File":"WJets.root",
        "color": "green",
        "latex": r"W + jets"
    },
    "DYJets":{
        "Use":True,
        "File":"DYJets.root",
        "color": "blue",
        "latex": r"DY + jets"
    },
    "Diboson":{
        "Use":True,
        "File":"Diboson.root",
        "color": "teal",
        "latex": r"VV"
    },
    "Electroweak":{
        "Use":True,
        "File":"Electroweak.root",
        "color": "green",
        "latex": r"W/DY+jets, VV",
    },
    "QCD":{
        "Use":True,
        "File":"QCD.root",
        "color": "dimgray",
        "latex": r"QCD"
    }
}
