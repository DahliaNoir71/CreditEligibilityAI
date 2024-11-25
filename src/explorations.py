import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt


def visualize_distribution(df_data):
    """
    Visualise la distribution des colonnes numériques d'un DataFrame.
    Affiche un histogramme et un boxplot pour chaque colonne, une figure par colonne.

    Paramètres:
        df_data (pd.DataFrame): Le DataFrame contenant les colonnes à visualiser.
    """
    numeric_columns = df_data.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_columns) == 0:
        print("Aucune colonne numérique trouvée à visualiser.")
        return

    for column in numeric_columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram with KDE
        sns.histplot(df_data[column], ax=axes[0], kde=True)
        axes[0].set_title(f"Histogramme avec KDE : {column}", fontsize=14)
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Fréquence")

        # Boxplot
        sns.boxplot(x=df_data[column], ax=axes[1], orient='h')
        axes[1].set_title(f"Boxplot : {column}", fontsize=14)
        axes[1].set_xlabel(column)

        plt.tight_layout()
        plt.show()


def visualize_categorical_distribution(df_data):
    """
    Visualise la distribution des colonnes catégorielles du DataFrame.
    """
    categorical_columns = df_data.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=column, data=df_data, hue=column, legend=False, palette="Set2")
        plt.title(f"Distribution de {column}")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.show()

def visualize_correlations(df_data, target_column=None):
    """
    Affiche la heatmap des corrélations entre toutes les colonnes numériques.
    Optionnellement, affiche la corrélation avec la colonne cible.
    """
    numeric_df = df_data.select_dtypes(include=['float64', 'int64'])

    # Corrélations entre toutes les colonnes
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Heatmap des corrélations entre toutes les colonnes numériques')
    plt.show()

    # Si une colonne cible est spécifiée, afficher les corrélations avec la cible
    if target_column and target_column in numeric_df.columns:
        correlation_with_target = numeric_df.corr()[target_column].sort_values(ascending=False)
        plt.figure(figsize=(6, len(correlation_with_target) * 0.5))
        sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(f'Heatmap des corrélations avec la cible : {target_column}')
        plt.show()



def visualize_missing_data(df_data):
    """
    Affiche la visualisation des valeurs manquantes pour chaque colonne.
    """
    msno.matrix(df_data)
    plt.show()


def visualize_target_correlation(df_data, target_column):
    """
    Visualise les relations entre une variable cible et les autres colonnes
    du DataFrame, en affichant des graphiques adaptés au type de données.

    Paramètres:
        df_data (pd.DataFrame): Le DataFrame contenant les données.
        target_column (str): Le nom de la colonne cible.
    """
    if target_column not in df_data.columns:
        print(f"La colonne cible '{target_column}' n'existe pas dans le DataFrame.")
        return

    # Vérifie si la cible est catégorique ou numérique
    is_target_categorical = df_data[target_column].dtype == 'object' or \
                            df_data[target_column].nunique() < 10

    for column in df_data.columns:
        if column == target_column:
            continue

        plt.figure(figsize=(10, 5))

        if df_data[column].dtype in ['float64', 'int64']:
            # Si la colonne est numérique
            if is_target_categorical:
                # Boxplot pour une cible catégorique
                sns.boxplot(data=df_data, x=target_column, y=column)
                plt.title(f"Boxplot de {column} par {target_column}")
            else:
                # Scatterplot pour une cible numérique
                sns.scatterplot(data=df_data, x=column, y=target_column)
                plt.title(f"Scatterplot entre {column} et {target_column}")
        else:
            # Si la colonne est catégorique
            if is_target_categorical:
                # Countplot pour une cible catégorique
                sns.countplot(data=df_data, x=column, hue=target_column)
                plt.title(f"Répartition de {column} par {target_column}")
            else:
                # Barplot pour une cible numérique
                sns.barplot(data=df_data, x=column, y=target_column, ci=None)
                plt.title(f"Moyenne de {target_column} par {column}")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def visualize_outliers(df_data):
    """
    Visualise les outliers dans les colonnes numériques du DataFrame
    à l'aide de boxplots.

    Paramètres:
        df_data (pd.DataFrame): Le DataFrame contenant les données.
    """
    numeric_columns = df_data.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_columns) == 0:
        print("Aucune colonne numérique trouvée pour l'analyse des outliers.")
        return

    for column in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df_data[column], orient='h',
                    flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5})
        plt.title(f"Détection des outliers pour {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.show()




def explore_dataframe(df_data, target_column):
    # Distribution des variables numériques
    visualize_distribution(df_data)

    # Visualisation des variables catégorielles
    visualize_categorical_distribution(df_data)

    # Corrélation entre toutes les variables
    visualize_correlations(df_data, target_column)

    # Visualisation des relations avec la variable cible
    visualize_target_correlation(df_data, target_column)

    # Visualisation des valeurs manquantes
    visualize_missing_data(df_data)

    # Identification des outliers
    visualize_outliers(df_data)
