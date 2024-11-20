import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.csv import get_data_from_csv
from src.optimizations import hyperparameter_search, get_best_models
from src.models import get_models
from src.param_grids import get_param_grids
from src.preprocessing import split_target_features, preprocess_data


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
    df_data = get_data_from_csv(csv_path, target, selected_features, column_id)
    train_data = preprocess_data(df_data, target)
    x, y = split_target_features(train_data, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Récupération des modèles et des grilles
    models = get_models()
    param_grids = get_param_grids()

    # Recherche des meilleurs modèles
    best_models_grid_search = get_best_models(models, param_grids, GridSearchCV, x_train, y_train)




if __name__ == '__main__':
    main()
