def get_param_grids():
    """Retourne les grilles d'hyperparamètres pour chaque modèle."""
    return {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        },
        "K-Nearest Neighbors": {
            'n_neighbors': range(1, 31),
            'weights': ['uniform', 'distance']
        },
        "Support Vector Machine": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10]
        },
        "XGBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9]
        }
    }
