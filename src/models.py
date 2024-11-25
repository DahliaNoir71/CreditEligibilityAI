from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def get_models():
    """Retourne un dictionnaire des modèles à tester."""
    return {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier()  # Importation de la bibliothèque SVM pour SVM
    }
