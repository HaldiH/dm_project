import numpy as np
import pandas as pd

def load_data(data, train_perc: float = 0.7, test_perc: float = 0.3):
    """
    Splits the dataset to train and test
    inputs:
         data
         properties

    output:
        - X_train: training smiles
        - X_test: test smiles
        - y_train: training properties
        - y_test: test properties
    """
    idx = np.random.RandomState(seed=4738).permutation(len(data))   # random permutation of line indices

    _data = data[idx]       # _data is permutation of data, columns are not permutated. 

    train_test_size = [0.7, 0.3]
    idx_train = int(len(_data) * train_test_size[0])
    train = _data[0:idx_train]
    test = _data[idx_train:]

    return train, test

def get_smiles_encodings(file_path):
    """
    Returns smiles, alphabet and length of largest molecule in SMILES given a file containing SMILES molecules.
    input:
         file with molecules. Column's name must be 'smiles'.
    output:
        - smiles encoding
        - smiles alphabet (character based)
        - longest smiles string
    """
    df = pd.read_csv(file_path)

    smiles_list = np.asanyarray(df.Chiral_Molecular_SMILES)
    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))


    return smiles_list, smiles_alphabet, largest_smiles_len


def smile_to_hot(smile, largest_smile_len, alphabet):
    """
     Convert a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)


def split_array(arr, frac: float):
    return np.array_split(arr, [np.floor(frac * arr.shape[0]).astype(int)], axis=0)