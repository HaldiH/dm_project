from utils import load_data
import numpy as np
import pandas as pd
import IPython
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
    remove_duplicates_: bool = True, in_favour_of_col: str = 'Energy_(kcal/mol)') -> pd.DataFrame:
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

    if remove_duplicates_:
        pruned_dataset.sort_values(axis=0, by=['Chiral_Molecular_SMILES', in_favour_of_col], ascending=True, inplace=True)
        pruned_dataset.drop_duplicates(subset='Chiral_Molecular_SMILES', keep='first', inplace=True)
    
    return pruned_dataset

def encode_smiles_column_of(dataset: pd.DataFrame, strategy: str = 'one_hot_encoding'):
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
    smiles = encoded_dataset['Chiral_Molecular_SMILES']

    if strategy == 'one_hot_encoding':
        encoded_dataset = pd.get_dummies(dataset, columns=['Chiral_Molecular_SMILES'])

    if strategy == 'count_encoding':
        smiles_alphabet = list(set(''.join(np.asanyarray(smiles))))
        for letter in smiles_alphabet:
            encoded_dataset[letter] = encoded_dataset.apply(lambda row: row['Chiral_Molecular_SMILES'].count(letter), axis=1)
        
        del encoded_dataset['Chiral_Molecular_SMILES']

    return encoded_dataset

def normalize_ndarray(X: np.ndarray):
    """Normalize every column of X by substracting mean and dividing by std"""
    for j in range(X.shape[1]):
        if np.std(X[:, j]) != 0: 
            X[:, j] = (X[:, j]-np.mean(X[:, j]))/np.std(X[:, j])
        else:
            X[:, j] = 0
    return X

def return_required_data(dataset: pd.DataFrame, targets: list[str], normalize=True): 
    target_indices = []
    for target in targets:
        target_indices.append(dataset.columns.get_loc(target))

    X_train, X_rest = load_data(np.asanyarray(dataset)) # splits 50/50
    X_val, X_test = load_data(np.asanyarray(X_rest), train_perc=0.8, test_perc=0.2) # e.g. 25% val, 25% test

    y_test = np.ndarray(shape=(X_test.shape[0], len(targets)))
    y_train = np.ndarray(shape=(X_train.shape[0], len(targets)))
    y_val = np.ndarray(shape=(X_val.shape[0], len(targets)))

    target_indices.sort(reverse=True)
    # remove target columns from X and add them to y
    # with train and test data
    for i, index in enumerate(target_indices):                
        y_train[:, i] = X_train[:, index]
        X_train = np.delete(X_train, index, 1)

        y_test[:, i] = X_test[:, index]
        X_test = np.delete(X_test, index, 1)

        y_val[:, i] = X_val[:, index]
        X_val = np.delete(X_val, index, 1)
    
    if normalize:
        X_train = normalize_ndarray(X_train)
        y_train = normalize_ndarray(y_train)
        
        X_val = normalize_ndarray(X_val)
        y_val = normalize_ndarray(y_val)
        
        X_test = normalize_ndarray(X_test)
        y_test = normalize_ndarray(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

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