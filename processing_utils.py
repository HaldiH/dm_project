from scipy import rand
from sklearn.preprocessing import StandardScaler
from utils import load_data
import numpy as np
import pandas as pd
import IPython
from typing import List, Optional
from sklearn.metrics import mean_squared_error

def ldf_display(df, nlines=500):
    """Utility function to display large dataframe portions as an html table."""
    txt = ("<iframe " +
           "srcdoc='" + df.head(nlines).to_html() + "' " +
           "width=1000 height=500 style=\"background: #FFFFFF;\">" +
           "</iframe>")
    return IPython.display.HTML(txt)

def prune_dataset_lines(
    dataset: pd.DataFrame, remove_nan_lines: bool = True, remove_nan_cols: bool = False, 
    remove_duplicates: bool = True, in_favour_of_col: str = 'Energy_(kcal/mol)') -> pd.DataFrame:
    """
    Remove lines from dataset if they contain nan values or are missing values.
    This function does not modify it's parameters.
    Lines with duplicate smiles will be dropped in favor of the one with lowest ```Energy_```, unless another
    column is provided.
    Lines with ```nan```/missing values are dropped. 

    Parameters
    -----------
    - dataset the dataset to prune. It is copied inside the function. 
    - remove_nan
    - remove_cols if specified, all columns with nan or empty values will be dropped.
    - remove_duplicates
    - in_favour_of_col the column on the which duplicate removal is done. 
    
    Returns
    -----------
    The pruned DataFrame by copy. 
    """
    pruned_dataset = dataset.copy()
    pruned_dataset.replace("nan", np.nan)
    
    if remove_nan_cols:
        pruned_dataset.dropna(axis=1, how='any', inplace=True)        

    if remove_nan_lines:  
        pruned_dataset.dropna(axis=0, how='any', inplace=True)

    if remove_duplicates:
        pruned_dataset.sort_values(axis=0, by=['Chiral_Molecular_SMILES', in_favour_of_col], ascending=True, inplace=True)
        pruned_dataset.drop_duplicates(subset='Chiral_Molecular_SMILES', keep='first', inplace=True)
    
    return pruned_dataset

def encode_smiles_column_of(dataset: pd.DataFrame, strategy: str = 'one_hot_encoding', column='Chiral_Molecular_SMILES'):
    """
    Encode smiles string column to numerical representation. 
    This function does not modify it's arguments.

    Parameters
    -----------
    - dataset the dataset to prune. It is copied inside the function. 
    - strategy one of 'one_hot_encoding', 'count_encoding'
    
    Returns
    -----------
    The encoded dataset by copy.
    """
    encoded_dataset = dataset.copy()
    smiles = encoded_dataset[column]
    encoded_dataset.drop(columns=column, inplace=True)

    if strategy == 'one_hot_encoding':
        encoded_dataset = pd.get_dummies(dataset, columns=column)

    if strategy == 'count_encoding':
        smiles_alphabet = np.unique(list(''.join(np.asanyarray(smiles))))
        for letter in str(smiles_alphabet):
            encoded_dataset[letter] = smiles.apply(lambda x: x.count(letter))
        
    return encoded_dataset


def normalize_ndarray(X: np.ndarray, continuous_columns: List[int]):
    """Normalize every column of X by substracting mean and dividing by std"""
    X = np.copy(X)
    for j in continuous_columns:
        column = X[:, j]
        std = column.std()
        if std != 0:
            X[:, j] = (column - column.mean()) / std
        else:
            X[:, j] = 0
    return X


def return_required_data(
        dataset: pd.DataFrame,
        targets: list[str],
        continuous_columns: list[str] = None,
        normalize=True,
        normalize_targets=True,
        validation=True,
        seed=4738
):
    n_rows = dataset.shape[0]

    # `sample` return a randomized copy of the DataFrame
    dataset = dataset.sample(random_state=np.random.RandomState(seed), frac=1)

    if normalize and continuous_columns is not None:
        mean = np.mean(dataset[continuous_columns], axis=0)
        std = np.mean(dataset[continuous_columns], axis=0)
        dataset[continuous_columns] = (
            dataset[continuous_columns] - mean) / std

    if normalize_targets:
        mean = np.mean(dataset[targets], axis=0)
        std = np.std(dataset[targets], axis=0)

        dataset[targets] = (dataset[targets] - mean) / std

    # train 50%, validation 25%, test 25%
    train_data, val_data, test_data = np.array_split(
        dataset, [int(1/2 * n_rows), int(3/4 * n_rows)])

    def split_feats_targs(df):
        feats_df = df.drop(columns=targets, inplace=False)
        targs_df = df.loc[:, targets]
        return feats_df, targs_df

    X_train, y_train = split_feats_targs(train_data)
    X_val, y_val = split_feats_targs(val_data)
    X_test, y_test = split_feats_targs(test_data)

    if validation:
        return np.asarray(X_train), np.asarray(y_train), np.asarray(X_val), np.asarray(y_val), np.asarray(X_test), np.asarray(y_test)
    else:
        return np.vstack([X_train, X_val]), np.vstack([y_train, y_val]), np.asarray(X_test), np.asarray(y_test)


def get_train_data(
        dataset: pd.DataFrame,
        targets_columns: list[str] = ['Energy_(kcal/mol)', 'Energy DG:kcal/mol)'],
        random_state: np.random.RandomState = None,
        validation=True,
        as_numpy=True,
):
    dataset = dataset.sample(frac=1, random_state=random_state)
    features_df: pd.DataFrame = dataset.drop(columns=targets_columns, inplace=False)
    targets_df: pd.DataFrame = dataset[targets_columns]
    n_rows = features_df.shape[0]

    # take 50%, 25%, 25% for validation data
    # else take 75%, 25%
    split_indices = (np.array([.5, .75] if validation else [.75]) * n_rows).astype(int)

    X_data: List[pd.DataFrame] = np.split(features_df, split_indices)
    y_data: List[pd.DataFrame] = np.split(targets_df, split_indices)

    t = ()
    for X, y in zip(X_data, y_data):
        t += (X.to_numpy(), y.to_numpy()) if as_numpy else (X, y)
    return t


def scale_data(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, X_val: pd.DataFrame=None, y_val: pd.DataFrame=None):
    """
    Scales training data by substracting mean and dividing by standard deviation.
    Centers the test data using the mean and standard deviation of the training data.
    """
    if X_val is not None and y_val is not None:
        val_f_scaler = StandardScaler()
        val_t_scaler = StandardScaler()
        test_f_scaler = StandardScaler()
        test_t_scaler = StandardScaler()

        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        X_train = val_f_scaler.fit_transform(X_train)
        y_train = val_t_scaler.fit_transform(y_train)
        X_val = val_f_scaler.transform(X_val)
        y_val = val_t_scaler.transform(y_val)

        X_train_val = test_f_scaler.fit_transform(X_train_val)
        y_train_val = test_t_scaler.fit_transform(y_train_val)
        X_test = test_f_scaler.transform(X_test)
        y_test = test_t_scaler.transform(y_test)

        return ((val_f_scaler, val_t_scaler), (test_f_scaler, test_t_scaler)), (X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test)
    else:
        f_scaler = StandardScaler()
        t_scaler = StandardScaler()
        X_train = f_scaler.fit_transform(X_train)
        y_train = t_scaler.fit_transform(y_train)
        X_test = f_scaler.transform(X_test)
        y_test = t_scaler.transform(y_test)
        return (f_scaler, t_scaler), (X_train, y_train, X_test, y_test)


def cross_validation_of(Algorithm, X: np.ndarray, y: np.ndarray, V: int = 10) -> float:
    """
    Compute cross validation error of Algorithm with X and y dataset and V partitions.

    Parameters
    -----------
    - Algorithm, a class containing a fit(X_train, y_train) and predict(X_train) method.
    - X the data without the targets, (you can np.vstack train and test data)
    - y the target(s) to predict. If multiple columns are specified the underlying Algorithm
      must be able to handle it. 
    
    Returns
    -----------
    the cross validation error
    """
    X_blocks = np.array_split(X, V, axis=0)
    y_blocks = np.array_split(y, V, axis=0)
    
    err = 0
    i = 0
    while i < len(X_blocks):
        X_test = X_blocks.pop(0)
        y_test = y_blocks.pop(0)
        
        Algorithm.fit(np.vstack(X_blocks), np.vstack(y_blocks))
        
        err += mean_squared_error(y_test, Algorithm.predict(X_test))
        
        X_blocks.append(X_test)
        y_blocks.append(y_test)

        i += 1

    return err / len(X_blocks)