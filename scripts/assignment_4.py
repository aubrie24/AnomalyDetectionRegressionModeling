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

# Custom transformer for anomaly detection
# Using Isolation Forest because it detects anomalies using random decision trees, which we learned about in class
class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.iforest = IsolationForest(contamination=self.contamination, random_state=42)
    
    def fit(self, X, y=None):
        self.iforest.fit(X)
        return self
    
    def transform(self, X):
        anomaly_scores = self.iforest.decision_function(X)

        # Avoid modifying the original dataset
        X = X.copy()

        # Ensure consistency without adding 'anomaly_score' to avoid mismatches during inference
        return X

# Function to load data from a text file
def load_txt(data_path):
    data = pd.read_csv(data_path, header=0, delimiter='\t')
    return data

# Function to separate features and target variable
def extract_features(data, label_col):
    labels = data[label_col]
    features = data.drop(label_col, axis=1)
    return features, labels

# Function to analyze which features have the highest coefficients for the models
def save_feature_importance(model, feature_names, model_name, importance_path):
    if hasattr(model, 'coef_'):
        importance = model.coef_
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"Feature importance not available for {model_name}")
        return

    # Ensure feature names and importance arrays are the same length
    if len(feature_names) != len(importance):
        print(f"Warning: Mismatch in feature name count and importance length for {model_name}. Skipping saving importance.")
        return
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    importance_df.to_csv(f"{importance_path}/{model_name}_feature_importance.csv", index=False)
    print(f"Feature importance for {model_name} saved.")

# Main function
def main():
    output_path = "../output/results.txt"
    model_path = "../output/modeling_pipeline.pkl"
    importance_path = "../output"
    predictions_path = "../output/ranked_predictions.csv"

    # The published data will be used to train and evaluate the model
    published_data_path = "/deac/csc/classes/csc373/data/assignment_4/published_screen.txt"

    # Load the published dataset
    published_data = load_txt(published_data_path)

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
        ('cap_infinite', CapInfiniteValues()),
        ('anomaly_detection', IsolationForestTransformer())
    ])

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

            # Extract processed feature names after preprocessing
            train_features_transformed = preprocessing_pipeline.transform(train_features)
            if isinstance(train_features_transformed, pd.DataFrame):
                processed_features = train_features_transformed.columns.tolist()
            else:
                processed_features = [f"feature_{i}" for i in range(train_features_transformed.shape[1])]

            save_feature_importance(model, processed_features, name, importance_path)

            if mse_test < best_mse:
                best_mse = mse_test
                best_model = pipeline
    
        joblib.dump(best_model, model_path)
        print("Model and results saved.")

# Function to test the trained model on unseen data
def test_on_unseen_data():
    model_path = "../output/modeling_pipeline.pkl"
    new_molecules_path = "/deac/csc/classes/csc373/data/assignment_4/new_molecules.csv"
    predictions_path = "../output/ranked_predictions.csv"

    # Load trained model
    model = joblib.load(model_path)
    print("Loaded trained model successfully.")

    # Load new data
    new_data = pd.read_csv(new_molecules_path)
    new_features = new_data.drop(columns=['id'], errors='ignore')

    # Make predictions using the full pipeline
    predictions = model.predict(new_features)

    # Save ranked predictions
    ranked_predictions = pd.DataFrame({
        'id': new_data['id'],
        'predicted_score': predictions
    }).sort_values(by='predicted_score', ascending=False)

    ranked_predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()
    test_on_unseen_data()
