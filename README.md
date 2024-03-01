# ProtSol

a novel approach to predict  the protein solubility based on the protein sequence.

## Files and folders:

**best_model:** the best performing model in paper

**checkpoint:** stores the best breakpoints in the training process.

**data:** the training set, test set, NESG set and Chang set used for training, testing and validation.

**model:** the BERT model uesd for ProtSol.

**Predict:** the folder used for prediction, please provide the fasta format file as in the example.

**record:** used to record the performance of the model during training.

**predict.py:** predicts the soluble tags corresponding to the amino acid sequences of the own dataset and outputs them to the file y_hat_own_data.csv.

**Solubilitylib.py:** some basic libraries for the model, mainly for data processing, both training and prediction will be used.

**train.py:** used for the model training process.

## 0. Environment

If you wanna run this model on your own linux servicer, plz git clone the ProtSol throuth the code below:

```shell
git clone https://github.com/binbinbinv/ProtSol.git
```

and then install the environment through the code below, but make sure you have installed the conda or miniconda on your servicer:

```shell
cd ProtSol/
conda env create environment.yml
```

and then run ProtSol in the ProtSol envirionment:

```shell
conda activate ProtSol
cd ProtSol/
```

Plz make sure you have downloaded the files in the folder:  
best_model
model


## 1. Predict your own sequences

Put your protein data in fasta format in the **./Predict/NEED_TO_PREPARE/own_data.fasta** folder, and make sure your data is in the same format as the example given.

Then run the command below:

```shell
python predict.py
```

You will get the output in ./Predict/Output.csv

## 2. Retrain the model

If you wanna retrain the model, plz run the code below:

```shell
python train.py
```
