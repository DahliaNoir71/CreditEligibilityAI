import pandas as pd


def get_data_from_csv(csv_path, target, features, column_id):
    try:
        # Read the CSV file into a DataFrame.
        df_data = pd.read_csv(csv_path)
        df_data = df_data.drop_duplicates(subset=[column_id])
        df_data = df_data[[target] + features]
        return df_data
    except FileNotFoundError:
        print("The specified CSV file does not exist.")
