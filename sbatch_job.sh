#!/bin/bash


### ENTER SLURM JOB COMMANDS HERE:

#SBATCH --partition=cms-uhh,cms,allgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL
#SBATCH --mail-user christopher.matthies@desy.de
#SBATCH --constraint=GPU
#SBATCH --chdir /beegfs/desy/user/matthies/WorkingArea/SingleTopClassifier/
#SBATCH -o ./outputs/job_outputs/slurm.%N.%j.out
#SBATCH -e ./outputs/job_outputs/slurm.%N.%j.err


### COMMANDS WHICH THE JOB CONSISTS OF:

source ~/.bashrc
source ~/conda.sh
cd /beegfs/desy/user/matthies/WorkingArea/SingleTopClassifier/

python ./TrainNeuralNetwork.py
