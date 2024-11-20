import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.optimizations import hyperparameter_search
from src.models import get_models
from src.param_grids import get_param_grids


def load_data(csv_path, target, selected_features, column_id):
    """Charge les données depuis un fichier CSV."""
    from src.csv import get_data_from_csv
    return get_data_from_csv(csv_path, target, selected_features, column_id)


def preprocess_data(df, target):
    """Prétraite les données : gère les NaNs et encode les variables catégoriques."""
    from src.preprocessing import categoricals_to_numeric, split_train_predict_data

    df = categoricals_to_numeric(df)  # Encode les variables catégoriques
    imputer = SimpleImputer(strategy="mean")  # Imputation des NaNs
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    train_data, predict_data = split_train_predict_data(df, target)
    return train_data


def split_features_target(train_data, target):
    """Sépare les features et la cible."""
    from src.preprocessing import split_target_features
    return split_target_features(train_data, target)


def build_model_pipeline(model, scaler=True):
    """Construit un pipeline avec option de standardisation."""
    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def main():
    # Paramètres globaux
    csv_path = "data/loan-data.csv"
    target = "Loan_Status"
    column_id = "Loan_ID"
    selected_features = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    # Chargement et prétraitement des données
    df_data = load_data(csv_path, target, selected_features, column_id)
    train_data = preprocess_data(df_data, target)
    x, y = split_features_target(train_data, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Récupération des modèles et des grilles
    models = get_models()
    param_grids = get_param_grids()

    # Création des pipelines
    pipelines = {
        name: build_model_pipeline(model, scaler=name != "Random Forest")
        for name, model in models.items()
    }

    # Recherche des meilleurs modèles
    best_models = hyperparameter_search(pipelines, param_grids, x_train, y_train)




if __name__ == '__main__':
    main()
