import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def infos(df_data):
    print("\nEXPLORE DATA")
    print(f"Shape : {df_data.shape}")
    display(df_data.info())
    display(df_data.head(10))
    display(df_data.columns)

def visualize_distribution(df_data):
    # Select only numeric columns.
    numeric_columns = df_data.select_dtypes(include=['float64', 'int64']).columns

    # Create a subplot for each numeric column.
    num_plots = len(numeric_columns)
    num_cols = 3  # Define the number of columns in the grid.
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed.

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    # Flatten the axes array for easy indexing, even if it's a 2D array.
    axes = axes.flatten()

    # Plot a histogram for each numeric column using sns.histplot.
    for i, column in enumerate(numeric_columns):
        sns.histplot(df_data[column], ax=axes[i], bins=30, kde=True)
        axes[i].set_title(column)

    # Hide any unused subplots.
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Adjust the spacing between subplots and the title.
    plt.tight_layout()  # Adjust top margin to accommodate the title.
    # Show the plot.
    plt.show()

def visualize_correlations(df_data, target_column):
    numeric_df = df_data.select_dtypes(include=['float64', 'int64'])
    correlation_with_target = numeric_df.corr()[target_column].sort_values(ascending=False)
    # Visualize the correlation matrix using sns.heatmap().
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_with_target, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Heatmap des corrélations avec {target_column}')
    plt.show()

def check_missing_values(df_data):
    print("\nCHECK MISSING VALUES")
    display(df_data.isnull().sum())
    # Calcul du pourcentage de valeurs manquantes
    missing_values_percentage = df_data.isnull().mean() * 100
    # Affichage des colonnes avec des valeurs manquantes supérieures à 5%
    missing_values_percentage = missing_values_percentage[missing_values_percentage > 5]
    print(missing_values_percentage)

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
