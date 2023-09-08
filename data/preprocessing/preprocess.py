import pandas as pd
import os

def load_data_global(path_data) :
    X_train = pd.read_csv(os.path.join(path_data, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(path_data, "X_test.csv"))
    Y_train = pd.read_csv(os.path.join(path_data, "Y_train.csv"))
    Y_test = pd.read_csv(os.path.join(path_data, "Y_test.csv"))
    return X_train, X_test, Y_train, Y_test

def load_data(path_data) :
    data = pd.read_csv(path_data)
    return data

# On remplace les NaN par des 0
def data_cleaning(data) : 
    data = data.fillna(0)
    return data

# Mapping global 
# NOTE : on choisit aussi de faire un embedding des valeurs chiffrées
def mapping(data) : 
    # Le pad est seulement présent pour mettre le futures valeurs inconnues
    # du jeu de test à 1
    val2idx = {0 : 0, 'UNK' : 1}  

    # Iterate through each row in the DataFrame
    for _, row in data.iterrows():
        for column, value in row.items():
            if value not in val2idx : 
                val2idx[value] = len(val2idx) + 1 

    return val2idx


def preprocessing_features(path_data) :
    X_train = data_cleaning(load_data(path_data))

    # Mapping global
    val2idx = mapping(X_train)
    
    # On remplace les valeurs par leur index
    X_train = X_train.applymap(lambda x : val2idx[x])

    return X_train, val2idx

def preprocessing_test_features(path_data, val2idx) :
    X_test = data_cleaning(load_data(path_data))

    # On doit prendre en compte les valeurs qui n'ont pas été vues dans le train
    # On les remplace par 1
    X_test = X_test.applymap(lambda x : val2idx.get(x, 1))

    return X_test

def preprocessing_labels(path_data) :
    Y_train = load_data(path_data)
    Y_train = Y_train.iloc[:, -1]
    return Y_train

def data_splitting(X_train, Y_train, test_size=0.2) :
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=test_size, random_state=42)
    return X_train, X_val, Y_train, Y_val


def column_selection(X_train) : 
    # On sélectionne les colonnes qui nous intéressent
    columns_items = [i for i in range(1,25)]
    columns_model_make = [i for i in range(49,97)]
    columns = columns_items + columns_model_make
    
    # On retourne les colonnes d'index compris dans columns
    return X_train.iloc[:,columns]


def preprocessing(path_data) : 
    
    # Load the dataset
    X_train_file = os.path.join(path_data, 'X_train.csv')
    y_train_file = os.path.join(path_data, 'Y_train.csv')

    X_train_df = pd.read_csv(X_train_file)
    y_train_df = pd.read_csv(y_train_file)

    cols_base = ['goods_code', 'model']
    colums_drop = ['ID'] + [col + str(i)
                            for col in cols_base for i in range(1, 25)]

    X_train_df = X_train_df.drop(colums_drop, axis=1)
    y_train_df = y_train_df['fraud_flag']

    # Identify the columns to apply RNN tokenization
    rnn_columns = ['item', 'make']  # Add more columns as needed
    rnn_columns = [col + str(i) for col in rnn_columns for i in range(1, 25)]

    # Identify the categorical and numerical columns
    categorical_columns = rnn_columns
    numerical_columns = [col for col in X_train_df.columns if col not in categorical_columns]

    # Clean data
    for col in categorical_columns:
        X_train_df[col] = X_train_df[col].fillna('')
    for col in numerical_columns:
        X_train_df[col] = X_train_df[col].fillna(0)

    return X_train_df, y_train_df, categorical_columns, numerical_columns