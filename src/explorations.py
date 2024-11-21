import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

def visualize_distribution(df_data, max_plots_per_figure=6):
    """
    Visualise la distribution des colonnes numériques du DataFrame.
    Crée plusieurs graphiques si le nombre de colonnes dépasse max_plots_per_figure.
    """
    numeric_columns = df_data.select_dtypes(include=['float64', 'int64']).columns
    num_plots = len(numeric_columns)

    if num_plots == 0:
        print("No numeric columns found to visualize.")
        return

    for start_idx in range(0, num_plots, max_plots_per_figure):
        subset_columns = numeric_columns[start_idx:start_idx + max_plots_per_figure]
        num_cols = 3
        num_rows = (len(subset_columns) + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
        axes = axes.flatten()

        for i, column in enumerate(subset_columns):
            sns.histplot(df_data[column], ax=axes[i], kde=True)
            sns.boxplot(x=df_data[column], ax=axes[i])
            axes[i].set_title(column, fontsize=12)

        for j in range(len(subset_columns), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

def visualize_categorical_distribution(df_data):
    """
    Visualise la distribution des colonnes catégorielles du DataFrame.
    """
    categorical_columns = df_data.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df_data[column], palette="Set2")
        plt.title(f"Distribution de {column}")
        plt.xticks(rotation=45)
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

    # Distribution de la variable cible
    sns.countplot(x=df_data[target_column])
    plt.title(f"Distribution de la variable cible : {target_column}")
    plt.show()
