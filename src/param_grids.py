from skopt.space import Real, Integer, Categorical

def get_param_grids():
    """Retourne les grilles d'hyperparamètres pour chaque modèle adaptées à la Bayesian Optimization."""
    return {
        "Logistic Regression": {
            'C': Real(0.001, 100, prior='log-uniform'),  # C est souvent optimisé logarithmiquement
            'penalty': Categorical(['l1', 'l2']),
            'solver': Categorical(['liblinear', 'saga']),
            'class_weight': Categorical([None, 'balanced'])
        },
        "K-Nearest Neighbors": {
            'n_neighbors': Integer(1, 30),  # Intervalle entier pour les voisins
            'weights': Categorical(['uniform', 'distance'])
        },
        "Random Forest": {
            'n_estimators': Integer(50, 500),  # Nombre d'arbres
            'max_depth': Integer(3, 50),  # Profondeur des arbres
            'min_samples_split': Integer(2, 10)  # Taille minimale des nœuds
        },
        "XGBoost": {
            'n_estimators': Integer(50, 300),  # Nombre d'estimateurs
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),  # Taux d'apprentissage
            'max_depth': Integer(3, 10)  # Profondeur maximale des arbres
        }
    }
