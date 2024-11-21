from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_search(model_name,
                          model,
                          search_method,
                          param_grid,
                          x_train_scaled,
                          y_train):
    print(f"{search_method.__name__} Optimization du modèle {model_name}...")
    search = None

    # Handling RandomizedSearchCV
    if search_method == RandomizedSearchCV:
        search = search_method(estimator=model,
                               param_distributions=param_grid,
                               n_iter=param_grid.get('n_iter', 10),  # Default n_iter if not present
                               cv=5,
                               scoring='accuracy',
                               verbose=1,
                               n_jobs=-1)

    # Fit the search object to the data
    search.fit(x_train_scaled, y_train)

    print(f"{search_method.__name__} Meilleur score pour {model_name} : {search.best_score_}")
    print(f"{search_method.__name__} Meilleurs hyperparamètres pour {model_name} : {search.best_params_}")

    # Access the best estimator
    best_estimator = search.best_estimator_

    return best_estimator


def get_best_models(models, param_grid, search_method, x_train_scaled, y_train):
    # Dictionnaire pour stocker les meilleurs modèles
    best_models = {}

    # Entraîner et optimiser chaque modèle
    for model_name, model in models.items():
        # Recherche du meilleur modèle
        best_model = hyperparameter_search(model_name,
                                           model,
                                           search_method,
                                           param_grid[model_name],
                                           x_train_scaled,
                                           y_train)

        # Enregistrer le meilleur modèle trouvé dans le dictionnaire
        best_models[model_name] = best_model

    return best_models