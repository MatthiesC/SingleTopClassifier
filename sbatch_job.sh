#!/bin/bash


### ENTER SLURM JOB COMMANDS HERE:

#SBATCH --partion=cms-uhh,cms,all-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL
#SBATCH --mail-user christopher.matthies@desy.de
#SBATCH --constraint=GPU
#SBATCH --workdir /beegfs/desy/user/matthies/WorkingArea/SingleTopClassifier/
#SBATCH -o ./output/job_outputs/slurm.%N.%j.out
#SBATCH -e ./output/job_outputs/slurm.%N.%j.err


### COMMANDS WHICH THE JOB CONSISTS OF:

source ~/.bashrc
source ~/conda.sh
cd /beegfs/desy/user/matthies/WorkingArea/SingleTopClassifier/

python ./TrainNeuralNetwork.py
