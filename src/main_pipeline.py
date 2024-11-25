import os
import joblib
from IPython.core.display_functions import display
from sklearn.model_selection import train_test_split
from src.csv import get_data_from_csv
from src.explorations import explore_dataframe
from src.preprocessing import split_train_predict, clean_loan_data, split_target_features, \
    transform_categorical_to_numeric


def prepare_data(csv_path, target, selected_features, column_id):
    """
    Charge et prépare les données pour l'entraînement et la prédiction.
    """
    df_data = get_data_from_csv(csv_path, target, selected_features, column_id)
    explore_dataframe(df_data, target)
    train_data, predict_data = split_train_predict(df_data, target)

    # Nettoyage des données
    train_data = clean_loan_data(train_data)
    predict_data = clean_loan_data(predict_data)

    # Appliquer la transformation catégorique sur les deux ensembles de données (entraînement et prédiction)
    train_data, label_encoders = transform_categorical_to_numeric(train_data)
    predict_data, _ = transform_categorical_to_numeric(predict_data, label_encoders)

    explore_dataframe(train_data, target)



    return train_data, predict_data, label_encoders



def split_and_train_data(train_data, target):
    """
    Sépare les données d'entraînement en X (features) et y (target) et les divise en ensembles d'entraînement et de test.
    """
    x, y = split_target_features(train_data, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def save_best_model(best_model, model_dir, model_name="xgboost_best_model.joblib"):
    """
    Sauvegarde le modèle entraîné dans un fichier joblib.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(best_model, model_path)
    print(f"Le meilleur modèle a été sauvegardé sous '{model_path}'.")
    return model_path


def predict_and_save(best_model, predict_data, selected_features, data_dir):
    """
    Effectue des prédictions avec le modèle et sauvegarde les résultats.
    """
    x_predict = predict_data[selected_features]
    predictions = best_model.predict(x_predict)
    predict_data['Predictions'] = predictions
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'loan_predictions.csv')
    predict_data.to_csv(csv_path, index=False)
    print(f"Les prédictions ont été sauvegardées dans {csv_path}.")
    display(predict_data)
    return csv_path