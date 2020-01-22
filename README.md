# SingleTopClassifier

Repository for DNN studies for the highly boosted tW measurement using Keras/TensorFlow


## Getting started

* You need to be on the `naf-cms-gpu01.desy.de` cluster and have Anaconda2 installed. Details [here](https://confluence.desy.de/pages/viewpage.action?spaceKey=UHHML&title=Using+GPUs+in+naf)
* Store UHH2 input ntuples into `samples` directory. This can be done with `. ./ReceiveNtuples.sh`
* Adjust `./config/InputDefinition.py` to your needs. It should be consistent with the `./src/DNNSetup.cxx` script in the [HighPtSingleTop repository](https://github.com/MatthiesC/HighPtSingleTop)
* Do `python GetInputs.py`. A `./workdir` will be created automatically if not existing. It will contain `.npy` files storing all possible DNN input variables as defined in `./config/InputDefinition.py`
* Adjust DNN hyperparameters and architecture in `TrainNeuralNetwork.py`
* Submit job to HTCondor via `condor_submit job.submit`

* Caveat: If you decided to use the Maxwell cluster instead... install your own conda environment as described on the page linked above and finally use `sbtach sbatch_job.sh`

## Other configuration

Configuration should be done within these files:
* `config/SampleClasses.py`: Specify which classes (physical processes like tW, TTbar, WJets etc.) you want to use for the multi-class DNN
* `config/InputDefinition.py`: Specify which input variables you want to use, i.e. how the DNN input vector looks like, and how to normalize each input variable. The templates for AK4 and HOTVR jets, the lepton, and the general event properties must be in accordance with `src/DNNSetup.cxx` within the `HighPtSingleTop` repository as mentioned before
