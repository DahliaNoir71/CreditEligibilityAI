import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def get_categorical_columns(df_data):
    # Colonnes catégorielles
    categorical_columns = df_data.select_dtypes(include=['object', 'category']).columns
    # Select binary categorical columns.
    binary_categorical_columns = [col for col in categorical_columns
                                  if df_data[col].dropna().nunique() == 2]
    # Select non binary categorical columns.
    other_categorical_columns = [col for col in categorical_columns
                                 if col not in binary_categorical_columns]

    return binary_categorical_columns, other_categorical_columns

def set_binary_to_numeric(binary_categorical_columns, df_data):
    # Convert binary categorical columns to numeric.
    for col in binary_categorical_columns:
        df_data.loc[df_data[col].notna(), col] = df_data.loc[df_data[col].notna(), col].map({df_data[col].unique()[0]: 0, df_data[col].unique()[1]: 1})
    return df_data


def set_non_binary_to_numeric(other_categorical_columns, df_data):
    for col in other_categorical_columns:
        if col == "Dependents":
            # Remplacer 3+ par 3
            df_data.loc[:, col] = df_data[col].str.replace("3+", "3")
        le = LabelEncoder()
        df_data.loc[:, col] = le.fit_transform(df_data[col].astype(str))
    return df_data


def categoricals_to_numeric(df_data):
    binary_categorical_columns, other_categorical_columns = get_categorical_columns(df_data)
    # Convert binary categorical columns to numeric.
    df_data = set_binary_to_numeric(binary_categorical_columns, df_data)
    # Convert non binary categorical columns to numeric.
    df_data = set_non_binary_to_numeric(other_categorical_columns, df_data)
    return df_data

def split_train_predict_data(df_data, target):
    # Séparer les données
    train_data = df_data[df_data[target].notna()]
    predict_data = df_data[df_data[target].isna()]
    return train_data, predict_data

def split_target_features(df_data, target):
    # Séparer les données
    x = df_data.drop(target, axis=1)
    y = df_data[target]
    return x, y



def preprocess_data(df, target):
    """Prétraite les données : gère les NaNs et encode les variables catégoriques."""
    df = df.dropna()
    df = categoricals_to_numeric(df)  # Encode les variables catégoriques
    train_data, predict_data = split_train_predict_data(df, target)
    return train_data


