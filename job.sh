#!/bin/bash

export PATH=/nfs/dust/cms/user/matthies/anaconda2/bin:$PATH
source activate /nfs/dust/cms/user/matthies/anaconda2/envs/myenv

PYTHONPATH=$PYTHONPATH:/nfs/dust/cms/user/matthies/anaconda2/envs/myenv/lib/python3.6/site-packages/
python3 /nfs/dust/cms/user/matthies/WorkingArea/SingleTopClassifier/TrainNeuralNetwork.py
