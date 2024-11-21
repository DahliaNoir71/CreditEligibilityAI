import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay,
                             accuracy_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             classification_report)


def calculate_metrics(y_test, y_pred):
    """
    Calcule les différentes métriques de performance pour un modèle.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


def display_classification_report(y_test, y_pred, model_name):
    """
    Affiche le rapport de classification pour un modèle.
    """
    print(f"Rapport de classification pour {model_name}:")
    print(classification_report(y_test, y_pred))


def display_confusion_matrix(model, x_test, y_test, model_name):
    """
    Affiche la matrice de confusion pour un modèle.
    """
    # Fermer toutes les figures précédentes avant d'en créer une nouvelle
    plt.close()

    # Création de la matrice de confusion
    cm_display = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)

    # Affichage avec des paramètres visuels personnalisés
    cm_display.plot(cmap='Blues', values_format='d')
    plt.title(f"Matrice de confusion pour {model_name}")
    plt.show()


def plot_comparison(results_df):
    """
    Crée et affiche un graphique comparatif des performances des modèles.
    """
    # Transformation du DataFrame pour faciliter l'affichage avec seaborn
    melted_results = results_df.melt(id_vars="Model",
                                     var_name="Metric",
                                     value_name="Value")

    # Création du graphique avec seaborn
    plt.figure(figsize=(10, 6))  # Taille du graphique
    sns.barplot(data=melted_results,
                x="Model",
                y="Value",
                hue="Metric",
                palette="viridis")  # Utilisation d'une palette de couleurs attractive

    # Personnalisation du graphique
    plt.title("Comparaison des performances des modèles", fontsize=16)
    plt.xlabel("Modèle", fontsize=12)
    plt.ylabel("Valeur des métriques", fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotation des labels des modèles pour les rendre lisibles
    plt.legend(title="Métrique", bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende à côté du graphique
    plt.tight_layout()  # Ajuste le layout pour éviter les coupures

    # Affichage du graphique
    plt.show()


def evaluate_models(best_models, x_test, y_test):
    """
    Évalue la performance des meilleurs modèles trouvés par GridSearchCV/RandomizedSearchCV
    sur le jeu de test.

    Args:
    - best_models: Dictionnaire contenant les meilleurs modèles après l'optimisation.
    - x_test: Données de test (caractéristiques).
    - y_test: Cibles de test.
    """
    # Liste pour stocker les résultats de chaque modèle
    results = []

    for model_name, model in best_models.items():
        # Prédictions du modèle sur les données de test
        y_pred = model.predict(x_test)

        # Calcul des différentes métriques
        accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

        # Affichage des résultats
        print(f"Performance du modèle {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Précision: {precision:.4f}")
        print(f"Rappel: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Affichage du rapport de classification
        display_classification_report(y_test, y_pred, model_name)

        # Stockage des résultats pour chaque modèle
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        # Affichage de la matrice de confusion
        display_confusion_matrix(model, x_test, y_test, model_name)

    # Création d'un DataFrame pour afficher les résultats
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1-Score", ascending=False)

    # Affichage des résultats sous forme de tableau
    print("Comparaison des modèles :")
    display(results_df)

    # Affichage du graphique comparatif des performances
    plot_comparison(results_df)
