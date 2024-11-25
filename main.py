
from skopt import BayesSearchCV
from src.evaluations import evaluate_models
from src.explorations import explore_dataframe
from src.main_pipeline import prepare_data, split_and_train_data, save_best_model, predict_and_save
from src.models import get_models
from src.optimizations import get_best_models
from src.param_grids import get_param_grids

def main():
    # Paramètres globaux
    csv_path = "data/loan-data.csv"
    target = "Loan_Status"
    column_id = "Loan_ID"  # Ajoutez cette ligne ici
    selected_features = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    model_dir = 'models'
    data_dir = 'data'

    # Préparation des données
    train_data, predict_data, label_encoders  = prepare_data(csv_path, target, selected_features, column_id)

    # Diviser les données d'entraînement et d'évaluation
    x_train, x_test, y_train, y_test = split_and_train_data(train_data, target)

    # Récupération des modèles et des grilles
    models = get_models()
    param_grids = get_param_grids()

    # Recherche des meilleurs modèles
    best_models = get_best_models(models, param_grids, BayesSearchCV, x_train, y_train)

    # Evaluation des modèles
    evaluate_models(best_models, x_test, y_test)

    # Sélection du meilleur modèle
    best_model = best_models['XGBoost']

    # Sauvegarde du meilleur modèle
    save_best_model(best_model, model_dir)

    predict_features = train_data.columns.drop('Loan_Status')

    # Prédictions et sauvegarde des résultats
    predict_and_save(best_model, predict_data, predict_features, data_dir)


if __name__ == '__main__':
    main()
