# SingleTopClassifier

A DNN for the highly boosted tW measurement using Keras/TensorFlow.


## Preliminaries

* Store UHH2 input ntuples into `samples` directory.
* A `workdir` directory will be created automatically if not existing. It will contain `.npy` files etc.


## DNN configuration

Configuration should be done within these files:
* `config/SampleClasses.py`: Specify which classes (physical processes like tW, TTbar, WJets etc.) you want to use for the multi-class DNN.
* `config/InputDefinition.py`: Specify which input variables you want to use, i.e. how the DNN input vector looks like.
