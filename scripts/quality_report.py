import sys, os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_file(file_path):
    #load CSV for TXT file
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, header=0)
        elif file_path.endswith('.txt'):
            return pd.read_csv(file_path, delimiter='\t', header=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def data_leak(data, target_column="SalePrice", corr_limit=0.9, report_file=None):
    with open(report_file, "a") as f:
        numeric_data = data.select_dtypes(include=['number'])
        categorical_data = data.select_dtypes(include=['object'])

        if target_column in numeric_data.columns:
            correlation_matrix = numeric_data.corr()
            sale_price_corr = correlation_matrix[target_column].abs().sort_values(ascending=False)
            potential_leak = sale_price_corr[sale_price_corr > corr_limit].index.tolist()
            if target_column in potential_leak:
                potential_leak.remove(target_column)

            if potential_leak:
                f.write(f"\nThe following numerical columns may cause data leakage due to correlation > {corr_limit}:\n {', '.join(potential_leak)}\n")
            else:
                f.write("\nNo numerical features were found to have a strong correlation with the target.\n")
        else:
            f.write(f"Warning: Target column '{target_column}' is not numeric or missing from data.\n")

        if not categorical_data.empty:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
            encoded_arr = encoder.fit_transform(categorical_data)
            encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(categorical_data.columns))
            encoded_df[target_column] = data[target_column]

            corr_matrix = encoded_df.corr()
            target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)

            categorical_leak = target_corr[target_corr > corr_limit].index.tolist()
            if target_column in categorical_leak:
                categorical_leak.remove(target_column)

            if categorical_leak:
                f.write(f"\nThe following categorical (OHE) columns may cause data leakage due to correlation > {corr_limit}:\n {', '.join(categorical_leak)}\n")
            else:
                f.write("\nNo categorical features were found to have a strong correlation with the target.\n")

def data_shape(data, report_file):
    with open(report_file, "a") as f:
        f.write(f"\nNumber of columns: {data.shape[1]}\n")
        f.write(f"Number of rows: {data.shape[0]}\n")

def duplicate_rows(data, report_file):
    with open(report_file, "a") as f:
        f.write(f"\nDuplicate rows: {len(data[data.duplicated()])}\n")

def rows_missing(data, report_file):
    with open(report_file, "a") as f:
        f.write(f"\nNumber of rows with missing values: {len(data[data.isna().any(axis=1)])}\n")

def col_missing(data, report_file):
    with open(report_file, "a") as f:
        missing_cols = data.columns[data.isna().any()].tolist()
        f.write(f"Columns that have missing values: {missing_cols}\n")
        f.write(f"Number of columns with missing values: {len(missing_cols)}\n")

def data_type(data, report_file):
    with open(report_file, "a") as f:
        f.write("\nData types of columns:\n")
        f.write(f"{data.dtypes.to_string()}\n")

def data_range(data, report_file):
    with open(report_file, "a") as f:
        numeric_data = data.select_dtypes(include=['number'])
        if not numeric_data.empty:
            f.write(f"\nMax values of numeric columns:\n{numeric_data.max().to_string()}\n")
            f.write(f"\nMin values of numeric columns:\n{numeric_data.min().to_string()}\n")

def unique_values(data, report_file):
    with open(report_file, "a") as f:
        cat_cols = data.select_dtypes(include=['object'])
        if not cat_cols.empty:
            f.write(f"\nUnique values for categorical columns:\n")
            for col in cat_cols.columns:
                unique_values = data[col].unique()
                if len(unique_values) <= 10:
                    f.write(f"{col} {unique_values}\n")
                else:
                    f.write(f"{col}: {unique_values[:10]}... (Total unique values: {len(unique_values)})\n")

def check_quality(file_path, output_dir):
    data = load_file(file_path)
    if data is None:
        return

    file_name = os.path.basename(file_path).replace('.csv', '').replace('.txt', '')
    output_file = os.path.join(output_dir, f"quality_report_{file_name}.txt")

    with open(output_file, "w") as f:
        f.write("Quality Report:\n")

    data_shape(data, output_file)
    duplicate_rows(data, output_file)
    rows_missing(data, output_file)
    col_missing(data, output_file)
    data_type(data, output_file)
    data_range(data, output_file)
    unique_values(data, output_file)
    data_leak(data, report_file=output_file)

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print('Enter path of directory containing csv/txt files.')
        sys.exit(0)
    elif not os.path.exists(args[0]):
        print('Invalid filepath of the directory')
        sys.exit(0)
    else:
        directory_path = args[0]
        print("Running data quality report for all CSV and TXT files.")

        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')

        for file in os.listdir(directory_path):
            if file.endswith(('.csv', '.txt')):
                file_path = os.path.join(directory_path, file)
                check_quality(file_path, output_dir)

        print(f"Data quality report saved to files in the output directory.")

if __name__ == "__main__":
    main()
