from IPython.core.display_functions import display
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from src.csv import get_data_from_csv
from src.models import get_models
from src.optimizations import get_best_models
from src.param_grids import get_param_grids
from src.preprocessing import split_train_predict, clean_loan_data, split_target_features


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
    train_data, predict_data = split_train_predict(df_data, target)
    train_data = clean_loan_data(train_data)
    x, y = split_target_features(train_data, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Récupération des modèles et des grilles
    models = get_models()
    param_grids = get_param_grids()

    # Recherche des meilleurs modèles
    best_models_grid_search = get_best_models(models, param_grids, RandomizedSearchCV, x_train, y_train)

    # Affichage des meilleurs modèles
    display(best_models_grid_search)


if __name__ == '__main__':
    main()
