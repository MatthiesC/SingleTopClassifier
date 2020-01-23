fileNamePrefix_MC = "samples/uhh2.AnalysisModuleRunner.MC."
fileNamePrefix_DATA = "samples/uhh2.AnalysisModuleRunner.DATA."

dict_Classes = {
    "tW_signal":{
        "Use":True,
        "File":"ST_tW_signal.root",
        "color": "blue",
        "latex": r"tW signal"
    },
    "tW_other":{
        "Use":True,
        "File":"ST_tW_other.root",
        "color": "cyan",
        "latex": r"tW bkg."
    },
    "tW_bkg_TopToHadAndWToTau":{
        "Use":True,
        "File":"ST_tW_bkg_TopToHadAndWToTau.root",
        "color": "deepskyblue",
        "latex": r"t(h)W($\tau}$) bkg."
    },
    "tW_bkg_Else":{
        "Use":True,
        "File":"ST_tW_bkg_Else.root",
        "color": "cyan",
        "latex": r"other tW bkg."
    },
    "tChannel":{
        "Use":True,
        "File":"ST_non-tW_t-Ch.root",
        "color": "orange",
        "latex": r"$t$-channel"
    },
    "sChannel":{
        "Use":True,
        "File":"ST_non-tW_s-Ch.root",
        "color": "brown",
        "latex": r"$s$-channel"
    },
    "TTbar":{
        "Use":True,
        "File":"TTbar.root",
        "color": "red",
        "latex": r"$\mathrm{t}\bar{\mathrm{t}}$"
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
        "color": "gold",
        "latex": r"DY + jets"
    },
    "Diboson":{
        "Use":True,
        "File":"Diboson.root",
        "color": "magenta",
        "latex": r"VV"
    },
    "QCD":{
        "Use":True,
        "File":"QCD.root",
        "color": "gray",
        "latex": r"QCD"
    }
}
