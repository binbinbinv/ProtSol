# ProtSol

A novel approach to predict  the protein solubility based on the protein sequence.

## Files and folders:

**best_model:** the best performing model in the paper.  

**checkpoint:** stores the best breakpoints during training.  

**data:**  Training set, test set, NESG set and Chang set for training, testing and validation.  

**model:** BERT model uesd for ProtSol.  

**log:** the folder where the training logs are kept.

**Predict:** folder for prediction, please provide the fasta format file in the example.  

**predict.py:** predict the solubility label corresponding to the amino acid sequence of your own dataset, and output it to y_hat_own_data.csv file.  

**Solubilitylib.py:**  some basic libraries for the model, mainly used for data processing, both for training and prediction.  

**train.py:** used for model training process.

## 0. Environment

If you want to run this model in a standalone mode on your own system, git-clone ProtSol with the following code:

```shell
git clone https://github.com/binbinbinv/ProtSol.git
```

Then install the ProtSol environment with the following commands, but first make sure you have conda or miniconda installed on your server:

```shell
cd ProtSol/
conda env create -f environment.yml
```

If the automatic installation using environment.yml fails, you may install ProtSol manually by following the instructions:

```shell
conda create -n ProtSol python=3.8
conda activate ProtSol
pip3 install torch torchvision torchaudio
pip install pandas bio seaborn matplotlib_inline
pip install scikit-learn transformers Ipython
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
