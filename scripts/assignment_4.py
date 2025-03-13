import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

#function to convert columns to numeric where possible
#this will have to be done before any modeling 
def convert_to_numeric(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def debug_infinite_values(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    for col in numeric_cols.columns:
        # Count how many inf or -inf in each column
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"Column '{col}' has {inf_count} infinite values.")

#function to handle missing values
#according to the quality report, almost all rows have missing values (cannot remove all rows)
#however, none of the columns have a significantly large portion of NAs (cannot remove valuable columns)
#so, impute the missing values
#function written by chat
def handle_missing_values(data, threshold=0.3, cap_value=1e9):
    # 1. Drop columns that are entirely NaN
    data = data.dropna(axis=1, how='all')

    # 2. Drop columns with missing values exceeding the threshold
    missing_ratio = data.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > threshold].index
    data = data.drop(columns=columns_to_drop)

    # 3. Replace inf and -inf with a large finite cap
    #    (Alternative: replace them with NaN, then impute)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = data[col].replace(np.inf, cap_value)
        data[col] = data[col].replace(-np.inf, -cap_value)

    # 4. Impute remaining missing values with column mean
    data = data.fillna(data.mean())

    # 5. Final safety check: Drop rows that still have NaN
    #    (in case columns were all inf or still not fixable by imputation)
    data = data.dropna()

    # Debug prints to confirm no inf or NaN remain
    print(f"Remaining NaN values: {data.isnull().sum().sum()}")
    print(f"Remaining Inf values: {np.isinf(data).sum().sum()}")
    return data

#function to load the data from a txt file 
def load_txt(data_path):
    data = pd.read_csv(data_path, header=0, delimiter='\t')
    data = convert_to_numeric(data) #change data types to numeric
    #drop the problematic 'gmin' column (this kept causing modeling to crash even after trying to impute or cap values)
    if 'gmin' in data.columns:
        data.drop(columns=['gmin'], inplace=True, errors='ignore')
    data = handle_missing_values(data) #input any missing values with the mean
    debug_infinite_values(data) #for debugging 
    return data

#function to seperate features and target variable
def extract_features(data, label_col):
    labels = data[label_col]
    features = data.drop(label_col, axis=1)
    return features, labels

#the baseline regressor: linear regression
#this will not work unless missing values are removed
def baseline_regressor(train_features, train_labels, test_features, test_labels):
    model = LinearRegression()
    model.fit(train_features, train_labels)
    train_predictions = model.predict(train_features) #predict on the same data that was used to train
    test_predictions = model.predict(test_features) #predict on data that was not used to train
    mse_train = mean_squared_error(train_labels, train_predictions) #evaluate MSE for train data
    mse_test = mean_squared_error(test_labels, test_predictions) #evaluate MSE for test data
    print(f"Baseline MSE for Train Set: {mse_train}")
    print(f"Baseline MSE for Test Set: {mse_test}")
    return model, mse_train, mse_test

# Random Forest Regressor
def random_forest_regressor(train_features, train_labels, test_features, test_labels):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_features, train_labels)
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    mse_train = mean_squared_error(train_labels, train_predictions)
    mse_test = mean_squared_error(test_labels, test_predictions)
    print(f"Random Forest MSE for Train Set: {mse_train}")
    print(f"Random Forest MSE for Test Set: {mse_test}")
    return model, mse_train, mse_test

# Gradient Boosting Regressor
def gradient_boosting_regressor(train_features, train_labels, test_features, test_labels):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(train_features, train_labels)
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    mse_train = mean_squared_error(train_labels, train_predictions)
    mse_test = mean_squared_error(test_labels, test_predictions)
    print(f"Gradient Boosting MSE for Train Set: {mse_train}")
    print(f"Gradient Boosting MSE for Test Set: {mse_test}")
    return model, mse_train, mse_test


def main():
    output_path = "../output/results.txt"
    model_path = "../output/modeling_pipeline.pkl"
    predictions_path = "../output/ranked_predictions.csv"

    #the published data will be used to train and evaluate the model
    published_data_path = "/deac/csc/classes/csc373/data/assignment_4/published_screen.txt"
    #the new molecules data will be used for predictions and output ranked data, but cannot be evaluated 
    new_molecules_path = "/deac/csc/classes/csc373/data/assignment_4/new_molecules.csv"

    #load the published dataset
    published_data = load_txt(published_data_path)

    #split the published dataset into train and dev
    train_data, test_data = train_test_split(published_data, test_size=0.2, random_state=42)

    #extract features and labels for the train and dev sets
    train_features, train_labels  = extract_features(train_data, "Inhibition")
    test_features, test_labels = extract_features(test_data, "Inhibition")

    #train and evaluate baseline regressor 
    bl_regressor, bl_train_mse, bl_test_mse = baseline_regressor(train_features, train_labels, test_features, test_labels)

    #train and evaluate random forest regressor
    rf_regressor, rf_train_mse, rf_test_mse = random_forest_regressor(train_features, train_labels, test_features, test_labels)

    #train and evaluate gradient boosting regressor
    gb_regressor, gb_train_mse, gb_test_mse = gradient_boosting_regressor(train_features, train_labels, test_features, test_labels)

    # Save results to CSV
    with open(output_path, "w") as file:
        file.write(f"Baseline MSE for Train Set: {bl_train_mse}\n")
        file.write(f"Baseline MSE for Test Set: {bl_test_mse}\n")
        file.write(f"Random Forest MSE for Train Set: {rf_train_mse}\n")
        file.write(f"Random Forest MSE for Test Set: {rf_test_mse}\n")
        file.write(f"Gradient Boosting MSE for Train Set: {gb_train_mse}\n")
        file.write(f"Gradient Boosting MSE for Test Set: {gb_test_mse}\n")
    print("Results saved to results.csv")


    if bl_test_mse <= rf_test_mse and bl_test_mse <= gb_test_mse:
        best_model = bl_regressor
    elif rf_test_mse <= bl_test_mse and rf_test_mse <= gb_test_mse:
        best_model = rf_regressor
    else:
        best_model = gb_regressor

    joblib.dump(best_model, model_path)

    
if __name__ == "__main__":
    main()
