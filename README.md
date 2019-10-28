# SingleTopClassifier

A DNN for the highly boosted tW measurement using Keras/TensorFlow.


## Preliminaries

* Store UHH2 input ntuples into `samples` directory. This done by `source ReceiveNtuples.sh`.
* Do `python GetInputs.py'. `A `workdir` directory will be created automatically if not existing. It will contain `.npy` files etc.


## DNN configuration

Configuration should be done within these files:
* `config/SampleClasses.py`: Specify which classes (physical processes like tW, TTbar, WJets etc.) you want to use for the multi-class DNN.
* `config/InputDefinition.py`: Specify which input variables you want to use, i.e. how the DNN input vector looks like. The templates for AK4 and HOTVR jets, the lepton, and the general event properties must be in accordance with `src/DNNSetup.cxx` within the `HighPtSingleTop` repository.
