import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def infos(df_data, with_objects=True):
    print("\nEXPLORE DATA")
    # Dimensions
    print(f"Shape : {df_data.shape}")
    # Types des données, colonnes non nulles
    print("\nInfo :")
    display(df_data.info())
    # Premières lignes
    print("\nHead :")
    display(df_data.head(10))
    # Dernières lignes
    print("\nTail :")
    display(df_data.tail(10))
    # Noms des colonnes
    print("\nColumns :")
    display(df_data.columns)
    # Résumé statistique des colonnes numériques
    print("\nStatistical Summary (Numeric Columns):")
    display(df_data.describe())
    if with_objects:
        # Résumé statistique des colonnes catégoriques
        print("\nStatistical Summary (Categorical Columns):")
        display(df_data.describe(include=['object', 'category']))
    # Vérification des valeurs manquantes
    print("\nMissing Values :")
    display(df_data.isnull().sum())
    # Calcul du pourcentage de valeurs manquantes
    missing_values_percentage = df_data.isnull().mean() * 100
    # Affichage des colonnes avec des valeurs manquantes supérieures à 5%
    missing_values_percentage = missing_values_percentage[missing_values_percentage > 5]
    print(missing_values_percentage)
    # Valeurs uniques par colonne
    print("\nUnique Values per Column:")
    display(df_data.nunique())
    # Aperçu des valeurs de chaque colonne
    print("\nColumn Value Counts:")
    for column in df_data.columns:
        print(f"\n{column}:")
        display(df_data[column].value_counts())
    # Nombre et pourcentage de NaN
    for column in df_data.columns:
        nan_count = df_data[column].isna().sum()
        total_count = len(df_data)
        print(f"{column} Nombre de NaN : {nan_count}, soit {nan_count / total_count:.2%} des données")


def visualize_distribution(df_data, max_plots_per_figure=6):
    """
    Visualizes the distribution of numeric columns in a DataFrame.
    Splits into multiple figures if there are more than max_plots_per_figure columns.

    Args:
        df_data (pd.DataFrame): The DataFrame containing the data.
        max_plots_per_figure (int): Maximum number of plots per figure.
    """
    # Select only numeric columns.
    numeric_columns = df_data.select_dtypes(include=['float64', 'int64']).columns
    num_plots = len(numeric_columns)

    if num_plots == 0:
        print("No numeric columns found to visualize.")
        return

    # Iterate over numeric columns and create plots in batches.
    for start_idx in range(0, num_plots, max_plots_per_figure):
        # Determine the subset of columns for this figure.
        subset_columns = numeric_columns[start_idx:start_idx + max_plots_per_figure]
        num_cols = 3  # Define the number of columns in the grid.
        num_rows = (len(subset_columns) + num_cols - 1) // num_cols  # Calculate rows.

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
        axes = axes.flatten()  # Flatten axes for easy indexing.

        # Create a histogram for each column in the current subset.
        for i, column in enumerate(subset_columns):
            sns.histplot(df_data[column], ax=axes[i], bins=30, kde=True)
            axes[i].set_title(column, fontsize=12)
            axes[i].set_xlabel("Value", fontsize=10)
            axes[i].set_ylabel("Frequency", fontsize=10)

        # Hide any unused axes in the grid.
        for j in range(len(subset_columns), len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and show the figure.
        plt.tight_layout()
        plt.show()

def visualize_correlations(df_data, target_column):
    """
    Visualise deux heatmaps :
    1. Corrélations entre toutes les colonnes numériques.
    2. Corrélations entre chaque colonne numérique et la colonne cible.

    Args:
        df_data (pd.DataFrame): Le DataFrame contenant les données.
        target_column (str): Le nom de la colonne cible.
    """
    # Sélectionner les colonnes numériques uniquement
    numeric_df = df_data.select_dtypes(include=['float64', 'int64'])

    # Vérifier si la colonne cible est numérique
    if target_column not in numeric_df.columns:
        raise ValueError(f"La colonne cible '{target_column}' n'est pas numérique ou absente des colonnes numériques.")

    # Calcul de la matrice de corrélation entre toutes les colonnes
    correlation_matrix = numeric_df.corr()

    # Calcul des corrélations avec la colonne cible
    correlation_with_target = correlation_matrix[target_column].sort_values(ascending=False)

    # **1. Heatmap des corrélations entre toutes les colonnes**
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Heatmap des corrélations entre toutes les colonnes numériques')
    plt.show()

    # **2. Heatmap des corrélations avec la colonne cible**
    plt.figure(figsize=(6, len(correlation_with_target) * 0.5))
    sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title(f'Heatmap des corrélations avec la cible : {target_column}')
    plt.show()

def visualize_outliers(df_data):
    # Select only numeric columns.
    numeric_columns = df_data.select_dtypes(include=['float64', 'int64']).columns

    # Create a subplot for each numeric column.
    num_plots = len(numeric_columns)
    num_cols = 3  # Define the number of columns in the grid.
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed.

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    # Flatten the axes array for easy indexing, even if it's a 2D array.
    axes = axes.flatten()

    # Plot a histogram for each numeric column using sns.boxplot.
    for i, column in enumerate(numeric_columns):
        sns.boxplot(df_data[column], ax=axes[i])
        axes[i].set_title(column)

    # Hide any unused subplots.
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Adjust the spacing between subplots.
    plt.tight_layout()

    # Show the plot.
    plt.show()

def visualize_dispersion(df_data):
    # Créer le pairplot
    sns.pairplot(df_data, hue='Loan_Status', diag_kind='kde', corner=False, palette='husl', height=2.5)

    # Afficher le graphique
    plt.show()

def visualize_all_plots(df_data, target_column):
    visualize_distribution(df_data)
    visualize_correlations(df_data, target_column)
    visualize_outliers(df_data)
    visualize_dispersion(df_data)
