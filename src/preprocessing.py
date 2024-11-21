import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder


def transform_categorical_to_numeric(df):
    """
    Transforme les colonnes catégoriques en numériques.
    Les colonnes binaires sont transformées en 0/1, et les colonnes non-binaires
    sont encodées avec LabelEncoder.

    Parameters:
    - df (pd.DataFrame): Le DataFrame contenant des colonnes catégoriques.

    Returns:
    - pd.DataFrame: Le DataFrame avec les colonnes catégoriques transformées.
    """
    # Identifier les colonnes catégoriques
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Séparer les binaires des non-binaires
    binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
    non_binary_cols = [col for col in categorical_cols if col not in binary_cols]

    # Transformer les colonnes binaires
    for col in binary_cols:
        df[col] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1}).astype(int)

    # Transformer les colonnes non-binaires
    label_encoder = LabelEncoder()
    for col in non_binary_cols:
        # Assurez-vous que la colonne est de type chaîne de caractères (str)
        df[col] = df[col].astype(str)
        df[col] = label_encoder.fit_transform(df[col])

    return df


def handle_missing_values(df):
    """
    Gère les valeurs manquantes dans un DataFrame.

    - Les colonnes catégoriques sont imputées avec la valeur la plus fréquente (mode).
    - Les colonnes numériques sont imputées avec la médiane.

    Parameters:
    - df (pd.DataFrame): Le DataFrame contenant des valeurs manquantes.

    Returns:
    - pd.DataFrame: Le DataFrame avec les NaN gérés.
    """
    # Identifier les colonnes catégoriques et numériques
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Imputation pour les colonnes catégoriques (valeur la plus fréquente)
    if len(categorical_cols) > 0:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = mode_imputer.fit_transform(df[categorical_cols])

    # Imputation pour les colonnes numériques (médiane)
    if len(numeric_cols) > 0:
        median_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = median_imputer.fit_transform(df[numeric_cols])

    return df


def clean_loan_data(df):
    # Encodage de 'Dependents' (transforme "3+" en 3 et encode)
    df.loc[df['Dependents'] != '3+', 'Dependents'] = df.loc[df['Dependents'] != '3+', 'Dependents'].fillna(
        df['Dependents'].mode()[0])
    df.loc[:, 'Dependents'] = df['Dependents'].replace('3+', 3)

    # One-hot encoding pour 'Property_Area'
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
    df = handle_missing_values(df)

    df = transform_categorical_to_numeric(df)

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
