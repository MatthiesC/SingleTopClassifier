#!/bin/bash

echo "Use scp? [y/n]:"

read answer_scp

declare -a samples=("ST_tW_DR_NoFullyHadronic_Sig" "ST_tW_DR_NoFullyHadronic_Bkg" "ST_otherChannels" "TTbar" "WJets" "DYJets" "Diboson" "QCD")

mkdir -p ./samples/run2/
for i in "${samples[@]}"
do
    if [ $answer_scp == "y" ]; then
	scp matthies@naf-cms:/nfs/dust/cms/user/matthies/HighPtSingleTop/102X_v2/mainsel/run2/both/nominal/hadded/uhh2.AnalysisModuleRunner.MC.${i}.root ./samples/run2/.
    else
	cp /nfs/dust/cms/user/matthies/HighPtSingleTop/102X_v2/mainsel/run2/both/nominal/hadded/uhh2.AnalysisModuleRunner.MC.${i}.root ./samples/run2/.
    fi
done
