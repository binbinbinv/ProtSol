# ProtSol

A novel approach to predict  the protein solubility based on the protein sequence.  

## Files and folders:  

**Predict:** folder for prediction, please provide the fasta format file in the example.  

**best_model:** stores the best-performing model in the paper, you should download the best model by following the README.md in this folder.  

**checkpoint:** stores the best breakpoints during training, if you retrain the ProtSol model, the checkpoint file will be in here.  

**data:**  stores the train set, test set, NESG set and Chang set for training, testing and validation, the data is in FASTA format and the labels have been contained in the description line by the '|' symbol, such as - ">2001test254|0" contains the label "0".  

**model:** stores the pre-trained model ProtTrans used for ProtSol, you should download the model files by following the README.md in this folder.  

**log:** the folder where logs during the training are kept.  

**predict.py:** Predict the solubility label corresponding to the amino acid sequence of your own dataset, and you can get the result file - Output.csv in the Predict folder.  

**Solubilitylib.py:**  stores some basic libraries for the model, mainly used for data processing, both for training and prediction.  

**train.py:** used for model training process.

## 0. Environment

If you want to run this model in a standalone mode on your own system, git-clone ProtSol with the following code:

```shell
git clone https://github.com/binbinbinv/ProtSol.git
```

Then install the ProtSol environment with the following commands, but first make sure you have conda or miniconda installed on your server.

Then you can install ProtSol environment manually by following the instructions:

```shell
conda create -n ProtSol python=3.8
conda activate ProtSol
pip install torch==2.2.2 torchvision torchaudio
pip install pandas bio seaborn matplotlib_inline
pip install scikit-learn transformers==4.39.3 Ipython
pip install iFeatureOmegaCLI rdkit
```

Then activate ProtSol's conda environment by issuing following command:

```shell
conda activate ProtSol
cd ProtSol/
```

To run the program, you need to go to each of the following two folders and follow the instructions in the Readme there to download and save the required data:  
**best_model**  
**model**

**To run predict.py and train.py on the GPU, make sure that the GPU has more than 24G of CUDA memory!** 

## 1. Predict your own sequences

Place your own protein sequence in fasta format in the folder of ./Predict/NEED_TO_PREPARE/own_data.fasta and make sure your data is in the same format as the example given.

Do not change the file name "own_data.fasta". If you have more than one sequence, put them all into one file named "own_data.fasta" as per the example provided.

Then run the following command:

```shell
python predict.py
```

predict.py can run on CPU and the predictions will be in the file ./Predict/Output.csv

## 2. Retrain the model

If you want to retrain the model, run the following code:

```shell
python train.py
```

The training process runs 40 epochs and it takes about 1 hour per epoch with nvidia 4090 24G.
