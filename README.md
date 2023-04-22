# ProtSol

a novel approach to predict  the protein solubility based on the protein sequence.

Files and folders:

best_model: stores the best-performing models.

checkpoint: stores the best breakpoints in the training process.

dataset: the training set and test set used for training.

own_dataset: the protein sequence used for prediction, please save it in csv and fasta format as in the example.

trainlog: used to record the performance of the model during training.

own_data_tesing.py: predicts the soluble tags corresponding to the amino acid sequences of the own dataset and outputs them to the file y_hat_own_data.csv.

Example:

```shell
python own_data_tesing.py . /own_dataset/
```

Solubilitylib.py: some basic libraries for the model, mainly for data processing, both training and prediction will be used.

train.py: used for the model training process.

