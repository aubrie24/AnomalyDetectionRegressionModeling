import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#function to convert columns to numeric where possible
#used in pipeline preprocessing
def convert_to_numeric(df):
    return df.apply(pd.to_numeric, errors='coerce')

#the gmin column kept making the code fail, no matter if values were imputted or clipped
#name is removed because this is not included in the un published data
#used in pipeline preprocessing
def drop_problem_columns(df):
    return df.drop(columns=['gmin', 'Name'], errors='ignore')

#custom transformer to handle missing values
#this transformer will drop any columns that has a high percentage of missing values
class MissingValues(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_drop = []
    
    def fit(self, X, y=None):
        #identify columns to drop based on missing ratio
        missing_ratio = X.isnull().mean()
        self.columns_to_drop = missing_ratio[missing_ratio > self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        #drop the identified columns
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        X = X.fillna(X.mean())
        return X
    
#custom transformer to cap infinite values
class CapInfiniteValues(BaseEstimator, TransformerMixin):
    def __init__(self, cap_value=1e9):
        self.cap_value = cap_value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], [self.cap_value, -self.cap_value])
        return X

#function to load the data from a txt file 
def load_txt(data_path):
    data = pd.read_csv(data_path, header=0, delimiter='\t')
    return data

#function to seperate features and target variable
def extract_features(data, label_col):
    labels = data[label_col]
    features = data.drop(label_col, axis=1)
    return features, labels

#function to analyze which features has the highest coefficients for the models
def save_feature_importance(model, feature_names, model_name, importance_path):
    if hasattr(model, 'coef_'):
        importance = model.coef_
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"Feature importance not available for {model_name}")
        return

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    importance_df.to_csv(f"{importance_path}/{model_name}_feature_importance.csv", index=False)
    print(f"Feature importance for {model_name} saved.")

def main():
    output_path = "../output/results.txt"
    model_path = "../output/modeling_pipeline.pkl"
    importance_path = "../output"
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

    #preprocessing steps
    preprocessing_pipeline = Pipeline(steps=[
        ('drop_problem_columns', FunctionTransformer(drop_problem_columns)),
        ('convert_numeric', FunctionTransformer(convert_to_numeric)),
        ('handle_missing', MissingValues()),
        ('cap_infinite', CapInfiniteValues())

    ])

    models = {
        "LinearRegression": LinearRegression(), #baseline
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
            file.write(f"{name} MSE for Test set:{mse_test}\n")

            # Extract processed feature names after preprocessing
            processed_features = preprocessing_pipeline.fit(train_features).transform(train_features).columns
            save_feature_importance(model, processed_features, name, importance_path)


            if mse_test < best_mse:
                best_mse = mse_test
                best_model = pipeline
            
        joblib.dump(best_model, model_path)
        print("Model and results saved.")

if __name__ == "__main__":
    main()
