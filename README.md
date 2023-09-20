# ProtSol

a novel approach to predict  the protein solubility based on the protein sequence.

## Files and folders:

**best_model:** the best performing model

**checkpoint:** stores the best breakpoints in the training process.

**dataset:** the training set and test set used for training.

**own_dataset:** the protein sequence used for prediction, please save it in csv and fasta format as in the example.

**trainlog:** used to record the performance of the model during training.

**own_data_tesing.py:** predicts the soluble tags corresponding to the amino acid sequences of the own dataset and outputs them to the file y_hat_own_data.csv.

**Solubilitylib.py:** some basic libraries for the model, mainly for data processing, both training and prediction will be used.

**train.py:** used for the model training process.

## 0. Environment

If you wanna run this model on your own linux servicer, plz git clone the ProtSol throuth the code below:

```shell
git clone git@github.com:binbinbinv/ProtSol.git
```

and then install the environment through the code below, but make sure you have installed the conda or miniconda on your servicer:

```shell
conda env create environment.yml
```

and then run ProtSol in the ProtSol envirionment:

```shell
conda activate ProtSol
cd ProtSol/
```

## 1. Test your own data

Example:

```shell
python own_data_tesing.py own_dataset/
```

Put your protein data in the **own_dataset** folder, and make sure your data is in the same format as the example given.

## 2. Retrain the model

If you wanna retrain the model, plz run the code below:

```shell
nohub python -u train.py > train.log 2>&1 &
```