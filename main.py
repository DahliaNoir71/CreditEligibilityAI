from src.csv import get_data_from_csv
from src.data_infos import infos, visualize_all_plots
from src.data_preprocessing import categoricals_to_numeric, split_target_data
from IPython.display import display


def main():

    csv_path = "data/loan-data.csv"
    target = "Loan_Status"
    column_id = "Loan_ID"
    selected_features = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    df_data = get_data_from_csv(csv_path, target, selected_features, column_id)
    train_data, predict_data = split_target_data(df_data, target)
    # infos(train_data, with_objects=True)
    train_data = categoricals_to_numeric(train_data)
    infos(train_data, with_objects=False)







if __name__ == '__main__':
    main()