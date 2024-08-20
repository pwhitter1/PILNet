# Code Repository for PIL-Net: Physics-Informed Graph Convolutional Network for Predicting Atomic Multipoles

### These codes allow for running the full PIL-Net machine learning pipeline on the QM Dataset for Atomic Multipoles.

## Dataset:

The QM Dataset for Atomic Multipoles is publicly accessible here: https://doi.org/10.3929/ethz-b-000509052.<br>
The `data.hdf5` and `data_gdb.hdf5` dataset files must be downloaded separately to run our experiments.<br>
Once downloaded, they can be placed in the `PILNet/src/datasets/` directory.<br>

Following this, you can run the sample program as `python run_example.py`, which will execute the full PIL-Net machine learning pipeline,
including data preprocessing, model training, and model inference.

## PIL-Net Files:

## Classes:

### 1. PILNet/src/neuralnetwork/PILNet.py
Contains class definition for PIL-Net model architecture.

### 2. PILNet/src/neuralnetwork/PILNet_Conv.py
Contains class definition for PIL-Net model convolutional layer.

## Machine Learning Pipeline:

### ** The main function of these files must be run from inside the `PILNet/` project directory, in the following numbered order. **

### 1. PILNet/src/preprocessing/ExtractFeatures.py
Read in dataset information, extract features, and save each molecule representation as a DGL graph.

### 2. PILNet/src/preprocessing/SplitData_NormalizeFeatures.py
Separate data into train/validation/test set splits and normalize features.

** ComputeReferenceMultipoles.py **

### 3. PILNet/src/training/TrainNetwork.py
Train a PILNet model using the training and validation dataset splits.

### 4. PILNet/src/inference/PredictNetwork.py
Use the trained PILNet model(s) to predict the test set labels.

## PIL-Net Models:
### PILNet/src/models/
Contains three trained PILNet models.<br>
Each model is trained on the same dataset split, but each uses a differently seeded randomization.
These models can be used for inference in lieu of running `TrainNetwork.py`.

`pilnet_model1.bin`<br>
`pilnet_model2.bin`<br>
`pilnet_model3.bin`<br>

## Command line arguments:

(NEED TO CHANGE THIS TO FUNCTION ARGUMENTS, NOT COMMAND LINE ARGUMENTS)

### Use the following arguments to specify data file paths when running the executable files.

### 1. ExtractFeatures.py
--read_filepath [Path to directory with hdf5 files containing QMDFAM data]<br>
--save_filepath [Path to directory to save DGL graphs of QMDFAM data]<br>

Example: `python ExtractFeatures.py  --read_filepath datasets/ --save_filepath graphs/`

### 2. SplitData_NormalizeFeatures.py
--read_filepath [Path to directory containing .bin files with DGL graphs of QMDFAM data]<br>
--save_filepath [Path to directory to save train/val/test splits of DGL graphs]<br>

Example: `python SplitData_NormalizeFeatures.py --read_filepath graphs/ --save_filepath splits/`

### 3. TrainNetwork.py
--read_filepath [Path to directory containing train/val/test splits of DGL graphs]<br>
--save_filepath [Path to directory to save trained pytorch model]<br>

Example: `python TrainNetwork.py --read_filepath splits/ --save_filepath models/`

### 4. PredictNetwork.py
--read_filepath_splits [Path to directory to containing train/val/test splits of DGL graphs]<br>
--read_filepath_model [Path to directory to trained pytorch model(s)]<br>

Example: `python PredictNetwork.py --read_filepath_splits splits/ --read_filepath_model models/`<br>

Note: All `.bin` files in the specified model directory will be used for model inference.
Their collective predictive error will be averaged.

## Code Dependencies:

### The following are the required libraries to run the codes and the corresponding versions we used.

`python==3.10.6`

`cuda==11.2.67`

`numpy==1.21.2`

`rdkit==2022.9.4`

`h5py==3.7.0`

`dgl==0.9.1 dgllife==0.3.2`

`torch==1.12.1 torchaudio==0.12.1 torchsummary==1.5.1 torchvision==0.13.1`

`psi4==1.9.1`