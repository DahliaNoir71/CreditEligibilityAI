import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder


def transform_categorical_to_numeric(df, label_encoders=None):
    """
    Transform categorical columns to numeric values.

    This function converts binary categorical columns to 0/1 and encodes non-binary
    categorical columns using LabelEncoder. It allows reusing encodings for predictions.

    Parameters:
    df (pd.DataFrame): The DataFrame containing categorical columns to be transformed.
    label_encoders (dict, optional): Dictionary of pre-existing encoders for non-binary columns.
                                     Defaults to None.

    Returns:
    tuple: A tuple containing two elements:
        - pd.DataFrame: The DataFrame with categorical columns transformed to numeric.
        - dict: Dictionary of LabelEncoders for each non-binary column (if not provided initially).

    """
    # Si aucun label_encoder n'est passé, on le crée
    if label_encoders is None:
        label_encoders = {}

    # Identifier les colonnes catégoriques
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Séparer les binaires des non-binaires
    binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
    non_binary_cols = [col for col in categorical_cols if col not in binary_cols]

    # Transformer les colonnes binaires
    for col in binary_cols:
        df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1}).astype(int)

    # Transformer les colonnes non-binaires
    for col in non_binary_cols:
        # Si l'encoder pour la colonne existe, l'appliquer
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))
        else:
            # Sinon, créer un nouveau LabelEncoder et l'appliquer
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le  # Sauvegarder l'encoder pour réutilisation future

    return df, label_encoders


def handle_missing_values(df):
    # Identifier les colonnes catégorielles et numériques
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(exclude=['object']).columns

    # Imputer les valeurs manquantes pour les colonnes catégorielles
    mode_imputer = SimpleImputer(strategy='most_frequent')

    # Appliquer l'imputation seulement aux colonnes non vides
    categorical_cols = [col for col in categorical_cols if df[col].notna().any()]

    if categorical_cols:  # Si des colonnes existent à imputer
        df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])

    # Imputer les valeurs manquantes pour les colonnes numériques
    mean_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = mean_imputer.fit_transform(df[numerical_cols])

    return df


def clean_loan_data(df):
    # Encodage de 'Dependents' (transforme "3+" en 3 et encode)
    df.loc[df['Dependents'] != '3+', 'Dependents'] = df.loc[df['Dependents'] != '3+', 'Dependents'].fillna(
        df['Dependents'].mode()[0])
    df.loc[:, 'Dependents'] = df['Dependents'].replace('3+', 3)

    # One-hot encoding pour 'Property_Area'
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
    df = handle_missing_values(df)

    return df


def split_train_predict(df, target):
    # Séparer les données avec Loan_Status défini et non défini
    df_train = df[df[target].notna()]
    df_predict = df[df[target].isna()]
    return df_train, df_predict


def split_target_features(df, target):
    x = df.drop(columns=[target])
    y = df[target]
    return x, y
