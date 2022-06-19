from typing import Tuple

import optuna
from optuna.trial import Trial

import pandas as pd
import numpy as np

from processing_utils import prune_dataset_lines, encode_smiles_column_of, return_required_data
from RBF import RBF

def objective(trial: Trial, data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    X_train, y_train, X_validation, y_validation = data
    n_clusters = trial.suggest_int("n_clusters", 1, 400)
    sigma = trial.suggest_float("sigma", .01, 10)
    rbf = RBF(n_clusters, sigma).fit(X_train, y_train)
    y_pred = rbf.predict(X_validation)
    mse = ((y_validation - y_pred)**2).mean()
    return mse

if __name__ == "__main__":
    raw_dataset = pd.read_csv('./dataset/data.csv')

    pruned_dataset = prune_dataset_lines(
        raw_dataset, remove_nan_lines=False, remove_nan_cols=True)
    print("Raw dataset shape =", raw_dataset.shape,
          " Pruned dataset shape =", pruned_dataset.shape)

    encoded_pruned_data = encode_smiles_column_of(
        pruned_dataset, 'count_encoding')  # change to one_hot_encoding here
    print("Encoded dataset shape =", encoded_pruned_data.shape)

    X_train_tot, y_train_tot, X_test, y_test = return_required_data(
        encoded_pruned_data,
        ['Energy_(kcal/mol)', 'Energy DG:kcal/mol)'],
        normalize=True
    )

    n_rows = X_train_tot.shape[0]
    t = np.floor(2/3 * n_rows).astype(int)

    X_train = X_train_tot[:t]
    y_train = y_train_tot[:t]
    X_val = X_train_tot[t:]
    y_val = y_train_tot[t:]

    study = optuna.create_study(
        direction="minimize", study_name="RBF hyperparameters optimization")
    study.optimize(lambda trial: objective(
        trial, (X_train, y_train, X_val, y_val)), n_trials=20)
    print(study.best_trial)

    n_clusters = study.best_params["n_clusters"]
    sigma = study.best_params["sigma"]
    rbf = RBF(n_clusters, sigma).fit(X_train_tot, y_train_tot)
    print("Model parameters: ", (rbf.lr.get_params()))