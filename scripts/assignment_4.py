import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import scipy.stats as stats

# Function to convert columns to numeric where possible
# Used in pipeline preprocessing
def convert_to_numeric(df):
    return df.apply(pd.to_numeric, errors='coerce')

# The gmin column kept making the code fail, no matter if values were imputed or clipped
# Name is removed because this is not included in the unpublished data
# Used in pipeline preprocessing
def drop_problem_columns(df):
    return df.drop(columns=['gmin', 'Name'], errors='ignore')

# Custom transformer to handle missing values
# This transformer will drop any columns that have a high percentage of missing values
class MissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_drop = []
    
    def fit(self, X, y=None):
        # Identify columns to drop based on missing ratio
        missing_ratio = X.isnull().mean()
        self.columns_to_drop = missing_ratio[missing_ratio > self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        # Drop the identified columns
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        X = X.fillna(X.mean())
        return X

# Custom transformer to cap infinite values
class CapInfiniteValues(BaseEstimator, TransformerMixin):
    def __init__(self, cap_value=1e9):
        self.cap_value = cap_value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], [self.cap_value, -self.cap_value])
        return X
'''
# Custom transformer for anomaly detection
# Using Isolation Forest because it detects anomalies using random decision trees, which we learned about in class
class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.iforest = IsolationForest(contamination=self.contamination, random_state=42)
        self.anomaly_mask = None
    
    def fit(self, X, y=None):
        self.iforest.fit(X)
        return self
    
    def transform(self, X):
        # Predict anomalies
        anomaly_predictions = self.iforest.predict(X)
        
        # Create a mask for non-anomalies (1 for inliers, -1 for outliers)
        self.anomaly_mask = anomaly_predictions == 1
        
        # Return only the inliers
        return X[self.anomaly_mask]
'''

# Function to load data from a text file
def load_txt(data_path):
    data = pd.read_csv(data_path, header=0, delimiter='\t')
    return data

# Function to separate features and target variable
def extract_features(data, label_col):
    labels = data[label_col]
    features = data.drop([label_col, 'CID'], axis=1, errors='ignore')
    return features, labels

# Function to detect anomalies and log them to a file
def detect_and_log_anomalies(features, cids, output_path, contamination=0.05):
    # Fit Isolation Forest
    iforest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_predictions = iforest.fit(features).predict(features)  # Use .predict for binary classification
    
    # Identify anomalies (predictions of -1 indicate anomalies)
    anomaly_mask = anomaly_predictions == -1
    
    # Create a DataFrame with CID and anomaly scores
    anomaly_report = pd.DataFrame({
        'CID': cids[anomaly_mask],
        'Anomaly_Score': anomaly_predictions[anomaly_mask]  # Use predictions for clarity
    }).sort_values(by='Anomaly_Score', ascending=True)
    
    # Save the report to a file
    anomaly_report.to_csv(output_path, index=False)
    print(f"Anomalies logged to {output_path}")

# Function to determine the best model based on the published data set
# Function to train and save the pipeline
def main():
    output_path = "../output/results.txt"
    model_path = "../output/modeling_pipeline.pkl"
    importance_path = "../output"
    predictions_path = "../output/ranked_predictions.csv"
    quality_report_path = "../output/quality_report.csv"

    # The published data will be used to train and evaluate the model
    published_data_path = "/deac/csc/classes/csc373/data/assignment_4/published_screen.txt"

    # Load the published dataset
    published_data = load_txt(published_data_path)

    ## Find the best model ##
    # Split the published dataset into train and dev
    train_data, test_data = train_test_split(published_data, test_size=0.2, random_state=42)

    # Extract features and labels for the train and dev sets
    train_features, train_labels = extract_features(train_data, "Inhibition")
    test_features, test_labels = extract_features(test_data, "Inhibition")

    # Preprocessing steps
    preprocessing_pipeline = Pipeline(steps=[
        ('drop_problem_columns', FunctionTransformer(drop_problem_columns)),
        ('convert_numeric', FunctionTransformer(convert_to_numeric)),
        ('handle_missing', MissingValues()),
        ('cap_infinite', CapInfiniteValues())
    ])

    train_features_transformed = preprocessing_pipeline.fit_transform(train_features)
    test_features_transformed = preprocessing_pipeline.transform(test_features)

    # Detect and log anomalies in the training data
    detect_and_log_anomalies(
        train_features_transformed,
        train_data['CID'].values,
        quality_report_path
    )

    models = {
        "LinearRegression": LinearRegression(),  # Baseline
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_mse = float('inf')

    with open(output_path, "w") as file:
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessing', preprocessing_pipeline),
                ('regressor', model)
            ])

            pipeline.fit(train_features, train_labels)
            train_preds = pipeline.predict(train_features)
            test_preds = pipeline.predict(test_features)

            mse_train = mean_squared_error(train_labels, train_preds)
            mse_test = mean_squared_error(test_labels, test_preds)

            file.write(f"{name} MSE for Train Set: {mse_train}\n")
            file.write(f"{name} MSE for Test Set: {mse_test}\n")

            if mse_test < best_mse:
                best_mse = mse_test
                best_model = pipeline
    
    ## Retrain data with the entire published dataset and the best model ##
    full_features, full_labels = extract_features(published_data, "Inhibition")

    # Create a new pipeline with the final model
    final_model = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('regressor', best_model)  # Use the best regressor
    ])

    final_model.fit(full_features, full_labels)

    # Save only the final model
    joblib.dump(final_model, model_path)
    print("Final model trained on full dataset and saved.")


# Function to test the trained model on unseen data
# Function to test trained model on new molecules
def test_on_unseen_data():
    model_path = "../output/modeling_pipeline.pkl"
    new_molecules_path = "/deac/csc/classes/csc373/data/assignment_4/new_molecules.csv"
    predictions_path = "../output/ranked_predictions.csv"

    model = joblib.load(model_path)
    new_data = pd.read_csv(new_molecules_path)
    new_data.rename(columns={'id': 'CID'}, inplace=True)

    # Apply the preprocessing pipeline to new data
    preprocessing_pipeline = model.named_steps['preprocessing']
    processed_features = preprocessing_pipeline.transform(new_data.drop(columns=['CID'], errors='ignore'))

    predictions = model.predict(processed_features)

    pd.DataFrame({'CID': new_data['CID'], 'predicted_score': predictions}) \
        .sort_values(by='predicted_score', ascending=False) \
        .to_csv(predictions_path, index=False)

    print(f"Predictions saved to {predictions_path}")

def feature_difference(ranked_path, features_path, model_path, output_path):
    # Load ranked predictions
    ranked_predictions = pd.read_csv(ranked_path)
    
    # Load new molecule feature data
    features_df = pd.read_csv(features_path)
    features_df.rename(columns={'id': 'CID'}, inplace=True)

    # Merge predictions with original feature data
    merged_df = ranked_predictions.merge(features_df, on='CID')

    # Load the trained model and extract the preprocessing pipeline
    model = joblib.load(model_path)
    preprocessing_pipeline = model.named_steps['preprocessing']

    # Extract features and apply full preprocessing
    feature_data = merged_df.drop(columns=['CID', 'predicted_score'], errors='ignore')
    processed_features = preprocessing_pipeline.transform(feature_data)

    # Convert processed features back to a DataFrame
    processed_df = pd.DataFrame(processed_features, columns=feature_data.columns)
    processed_df['CID'] = merged_df['CID'].values
    processed_df['predicted_score'] = merged_df['predicted_score'].values

    # Select top 100 and bottom 100 ranked molecules after preprocessing
    top_molecules = processed_df.nlargest(100, 'predicted_score')
    bottom_molecules = processed_df.nsmallest(100, 'predicted_score')

    # Drop non-feature columns (CID, predicted_score)
    feature_columns = [col for col in processed_df.columns if col not in ['CID', 'predicted_score']]

    # Store significant features
    significant_features = []

    # Perform statistical tests on cleaned and processed data
    for feature in feature_columns:
        top_values = top_molecules[feature].dropna()
        bottom_values = bottom_molecules[feature].dropna()

        if len(top_values) > 1 and len(bottom_values) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(top_values, bottom_values, equal_var=False)
                if p_value < 0.05:
                    significant_features.append((feature, p_value))
            except Exception as e:
                print(f"Skipping feature {feature} due to error: {e}")

    # Convert results to DataFrame and save
    significant_df = pd.DataFrame(significant_features, columns=['Feature', 'p_value']).sort_values(by='p_value')
    significant_df.to_csv(f"{output_path}/significant_features.csv", index=False)
    
    print(f"Significant feature analysis complete. Results saved to '{output_path}/significant_features.csv'.")

if __name__ == "__main__":
    main()
    test_on_unseen_data()

    # Define file paths
    predictions_path = "../output/ranked_predictions.csv"
    features_path = "/deac/csc/classes/csc373/data/assignment_4/new_molecules.csv"
    model_path = "../output/modeling_pipeline.pkl"
    output_path = "../output"

    # Run feature analysis with full preprocessing
    feature_difference(predictions_path, features_path, model_path, output_path)
