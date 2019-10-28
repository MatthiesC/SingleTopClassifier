#!/bin/bash

mkdir -p ./samples

declare -a arr=("ST_tW_signal" "ST_tW_other" "ST_non-tW_t-Ch" "ST_non-tW_s-Ch" "TTbar" "WJets" "DYJets" "Diboson" "QCD")

for i in "${arr[@]}"
do
    scp matthies@naf-cms:/nfs/dust/cms/user/matthies/HighPtSingleTop/102X_v1/MainSelection/2016/Muon/uhh2.AnalysisModuleRunner.MC.${i}.root ./samples/.
done
