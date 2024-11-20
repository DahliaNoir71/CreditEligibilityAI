from src.csv import get_data_from_csv
from src.data_infos import (infos,
                            check_missing_values,
                            visualize_all_plots)
from src.data_preprocessing import categoricals_to_numeric


def main():

    csv_path = "data/loan-data.csv"
    target = "Loan_Status"
    selected_features = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    df_data = get_data_from_csv(csv_path, target, selected_features)
    infos(df_data)
    check_missing_values(df_data)
    df_data = categoricals_to_numeric(df_data)
    visualize_all_plots(df_data, target)

    infos(df_data)





if __name__ == '__main__':
    main()