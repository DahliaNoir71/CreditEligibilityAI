from sklearn.preprocessing import LabelEncoder

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
        df_data[col] = df_data[col].map({df_data[col].unique()[0]: 0, df_data[col].unique()[1]: 1})

    return df_data
def set_non_binary_to_numeric(other_categorical_columns, df_data):
    # Encodage avec LabelEncoder
    label_encoders = {}
    for col in other_categorical_columns:
        le = LabelEncoder()
        # Convertir en string pour éviter les problèmes avec NaN
        df_data[col] = le.fit_transform(df_data[col].astype(str))
        # Sauvegarder le label encoder pour décodeur si nécessaire
        label_encoders[col] = le
    return df_data

def categoricals_to_numeric(df_data):
    binary_categorical_columns, other_categorical_columns = get_categorical_columns(df_data)
    # Convert binary categorical columns to numeric.
    df_data = set_binary_to_numeric(binary_categorical_columns, df_data)
    # Convert non binary categorical columns to numeric.
    df_data = set_non_binary_to_numeric(other_categorical_columns, df_data)

    return df_data