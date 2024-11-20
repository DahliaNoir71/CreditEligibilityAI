def get_param_grids():
    """Retourne les grilles d'hyperparamètres pour chaque modèle."""
    return {
        "Logistic Regression": {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__class_weight': [None, 'balanced']
        },
        "K-Nearest Neighbors": {
            'model__n_neighbors': range(1, 31),
            'model__weights': ['uniform', 'distance']
        },
        "Support Vector Machine": {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'poly', 'rbf'],
            'model__gamma': ['scale', 'auto']
        },
        "Random Forest": {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [None, 10, 20, 50],
            'model__min_samples_split': [2, 5, 10]
        },
        "XGBoost": {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__max_depth': [3, 6, 9]
        }
    }
